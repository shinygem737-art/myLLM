import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN     # 调用Hugging Face中的激活函数，如relu、gelu等
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class MyLLMConfig(PretrainedConfig):
    model_type = "myLLM"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        self.hidden_act = kwargs.get("hidden_act", 'silu')

        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)    # 向上取整到64的倍数，获得更好的计算效率

        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    # RMS: 对机制、大数值更敏感
    def __init__(self, dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # torch.rsqrt为开方求倒数
        # 在特征维度归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e5, rope_scaling: dict = None):
    # 处理YaRN相关部分
    # theta_d = rope_base ^ (-2 * d / D), d = 0, 1, 2, ..., D // 2 - 1
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            # beta_fast: beta   若L / lambda_d > beta  , 高频，只外推，lambda_d = 2 * pi / theta_d 波长
            # beta_slow: alpha  若L / lambda_d < alpha , 低频，只内插
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:    # L' > L 情况
            # 给定波长，找索引（这里找的是波长等于alpha或beta时的索引） 即通过L / lambda_d = alpha或beta, 反解lambda_d对应的索引d
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)  # 线性斜坡
            # (1 - ramp) * theta_d / s + ramp * theta_d = theta_d * (1 - ramp + ramp / s)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(0, end, device=freqs.device)
    # a = [a0, a1, ...]
    # b = [b0, b1, ...]
    # outer(a, b) = [[a0b0, a0b1, ...], [a1b0, a1b1, ...]]
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor   # (end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # RoPE的部分
    # rotate_half: [a, b, c, d] -> [-b, a, -d, c]. 这里我改了一下实现方式，变成相邻两个旋转了
    def rotate_half(x): 
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # GQA   Grouped-Query Attention 多个Q头共享KV头
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :]     # 插入一个大小为1的维度
            # expand不是复制数据，而是让多个索引位置共享同一块内存
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim) # 广播复制
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, config: MyLLMConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape 
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 在seq_len维度拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)    # 训练时开启dropout
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)     # scores: (bsz, n_local_heads, seq_len, seq_len)

            # 因果掩码
            if self.is_causal:  
                # 这里为什么不直接加？
                # (广播从最后一个维度向前匹配)
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)

            # padding 掩码
            if attention_mask is not None:
                # attention_mask: (bsz, seq_len) 所以要unsqueeze
                # attention_mask 值为1则为有效，为0则为无效
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax(scores): (bsz, n_local_heads, seq_len, seq_len)
            # softmax 是沿着key的维度做的
            # xv: (bsz, n_local_heads, seq_len, head_dim)
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MyLLMConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)   # 右侧Linear
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)     # 左侧Linear
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)   # 上面Linear
        self.act_fn = ACT2FN[config.hidden_act]
        self.drouput = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drouput(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
    

class MOEFeedForward(nn.Module):
    pass
    

class MyLLMBlock(nn.Module):
    def __init__(self, layer_id: int, config: MyLLMConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, 
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value        
    

class MyLLMModel(nn.Module):
    def __init__(self, config: MyLLMConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MyLLMBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)

        # 在 PyTorch 中，register_buffer 用于将张量注册为模块的缓冲区, 使其成为模型状态的一部分（会随模型保存和加载），但不作为可训练参数（即不会在反向传播中更新梯度）
        # persistent=False: 保存模型时不会保存该张量
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, 'layers'):  # ??
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # past_key_values[0]: (k_cache, v_cache)
        # k_cache: (bsz, history_len, num_kv_heads, head_dim)
        # v_cache: (bsz, history_len, num_kv_heads, head_dim)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])

        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MyLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MyLLMConfig
    def __init__(self, config: MyLLMConfig = None):
        self.config = config or MyLLMConfig()
        super().__init__(self.config)
        self.model = MyLLMModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享——让两个weight指向同一对象，减少参数量
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        # 生成最后logits_to_keep个元素的切片
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # cross_entropy内部自带softmax
            # x.view(-1, x.size(-1)): (bsz * (seq_len - 1), vocab_size)
            # ignore_index 指定一个被忽略的标签值（不参与损失计算）
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)