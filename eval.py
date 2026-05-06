import torch
import argparse
import time
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from model.model_myLLM import MyLLMConfig, MyLLMForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MyLLMForCausalLM(MyLLMConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            # 注入LoRA结构
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # 若load_from不是本地目录，则自动下载
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MyLLM模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    ## 
    parser.add_argument('--eval_ppl', default=False, action='store_true', help="评估困惑度（使用内置测试句子）")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []       # 存储多轮对话的历史记录
    model, tokenizer = init_model(args)
    # 流式输出器，实时打印生成的token，并跳过prompt和特殊token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    #   -----------测试ppl--------------------
    if args.eval_ppl:
        # 用独立的变量，避免覆盖全局 prompts
        ppl_test_prompts = [ '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
        test_texts = []
        for prompt in ppl_test_prompts:
            conversation = [{"role": "user", "content": prompt}]
            # 这里只构造 user 部分，不加 assistant 生成（因为我们要评估模型对 user 输入的似然）
            # 如果想让模型也评估自己的回答，需要提供完整对话
            formatted = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False   # 只评估已有文本
            )
            test_texts.append(formatted)
            ppl = compute_perplexity(model, tokenizer, [formatted])
            print(f"Prompt: {prompt}  |  Perplexity: {ppl:.4f}")
        return
    #   --------------------------------------


    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            # pretrain模式下直接把bos + prompt作为输入
            inputs = tokenizer.bos_token + prompt
        else:
            # 其他模式把conversation转化为聊天模板
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        # 调用generate方法生成回复
        # model的两个基类PreTrainedModel, GenerationMixin
        # PreTrainModel提供模型加载、保存、参数初始化等
        # GenerationMixin提供多种文本生成方法generate()，如贪心搜索、束搜索、采样(top_p)等
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # 将assistant的回复加入conversation历史中
        conversation.append({"role": "assistant", "content": response})
        # 生成token数量
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        # 速度显示
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')




def compute_perplexity(model, tokenizer, texts, stride=None, max_length=None, device=None):
    """
    计算一个语言模型在给定文本上的困惑度（Perplexity）。
    
    参数说明：
        model: HuggingFace 格式的语言模型（如 GPT, Llama 等），应具有 logits 输出。
        tokenizer: 对应的分词器，用于将文本转为 token ids。
        texts: 字符串或字符串列表，需要评估的原始文本。
        stride: 滑动窗口的步长。如果为 None，则设为 max_length//2。
        max_length: 模型一次能处理的最大 token 长度。若为 None，则从 model.config 读取。
        device: 计算设备，若为 None 则自动从模型参数中获取。
    
    返回：
        float: 困惑度值，如果所有文本长度均不足2则返回无穷大。
    """
    import torch
    import torch.nn.functional as F
    #from tqdm import tqdm

    model.eval()  # 切换为评估模式（关闭 dropout 等）

    # 确定计算设备
    if device is None:
        device = next(model.parameters()).device

    # 确定最大上下文长度（模型支持的最大 token 数）
    if max_length is None:
        # 尝试从常见配置字段中获取
        max_length = getattr(model.config, 'max_position_embeddings',
                             getattr(model.config, 'max_seq_len', 512))
    if max_length < 2:
        raise ValueError("max_length 必须 >= 2，否则无法计算 next-token loss。")

    # 设置滑动步长，默认取 max_length 的一半，保证相邻窗口有重叠
    if stride is None:
        stride = max_length // 2
    # 确保步长至少为1，且不超过 max_length-1（避免窗口无法滑动）
    stride = max(1, min(stride, max_length - 1))

    # 统一将输入转换为列表形式
    if isinstance(texts, str):
        texts = [texts]

    total_nll = 0.0   # 累积所有有效的负对数似然之和（自然对数形式）
    total_tokens = 0  # 累积所有参与损失计算的 token 总数

    # 逐条文本处理，使用 tqdm 显示进度
    #for text in tqdm(texts, desc="Computing Perplexity"):
    for text in texts:
        # 将文本编码为 token ids，不做截断，保留完整序列
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)   # shape: (1, seq_len)
        seq_len = input_ids.size(1)

        # 如果序列长度小于2，无法预测下一个 token（没有足够的位置），跳过该文本
        if seq_len < 2:
            continue

        start = 0  # 当前窗口的起始位置（包含）
        while start < seq_len - 1:  # 确保窗口至少包含一个预测位置（最后一个 token 作为 label）
            # 窗口的结束位置（不包含），不超过序列长度
            end = min(start + max_length, seq_len)
            input_window = input_ids[:, start:end]   # shape: (1, L)

            # 创建 labels：模型需要预测每个位置的下一个 token
            labels = input_window.clone()

            # 对于非起始窗口（即 start != 0），窗口前半部分与上一个窗口重叠。
            # 这些重叠 token 的损失已经在前一个窗口计算过了，所以忽略它们。
            # 只保留本窗口新增加的 token（即右侧末尾的 stride 个 token）参与损失。
            if start != 0:
                overlap = max(0, input_window.size(1) - stride)  # 重叠部分的长度
                if overlap > 0:
                    # 将重叠位置的 label 设为 -100，CrossEntropyLoss 将自动忽略它们
                    labels[:, :overlap] = -100

            # 模型前向推理，不计算梯度（节省内存）
            with torch.no_grad():
                outputs = model(input_window, use_cache=False)
                # logits shape: (1, L, vocab_size)
                # 预测下一个 token 的 logits：取前 L-1 个位置（因为最后一个位置没有“下一个 token”）
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                # 对应的 labels：取后 L-1 个位置
                shift_labels = labels[..., 1:].contiguous()

                # 计算每个位置的交叉熵损失（不对损失取平均，保留逐一损失值）
                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),  # 展平为 (N, vocab_size)
                    shift_labels.view(-1),                         # 展平为 (N,)
                    ignore_index=-100,                             # 忽略标记为 -100 的位置
                    reduction='none'                               # 返回每个 token 的损失
                )
                # 有效 token 数量（labels 中不为 -100 的位置）
                valid_mask = shift_labels.view(-1) != -100
                num_valid = valid_mask.sum().item()
                # 求和得到该窗口的总负对数似然（自然对数形式，因为 CrossEntropyLoss 默认用自然对数）
                window_nll = token_losses[valid_mask].sum().item() if num_valid > 0 else 0.0

            if num_valid > 0:
                total_nll += window_nll
                total_tokens += num_valid

            # 滑动窗口，步长为 stride
            start += stride

    # 如果没有计算到任何有效 token，返回无穷大（表示模型不能计算困惑度）
    if total_tokens == 0:
        return float('inf')

    # 平均负对数似然（自然对数）
    avg_nll = total_nll / total_tokens
    # 困惑度 = exp(平均负对数似然)
    return torch.exp(torch.tensor(avg_nll)).item()


if __name__ == "__main__":
    main()