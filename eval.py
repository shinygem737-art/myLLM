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
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # 流式输出器，实时打印生成的token，并跳过prompt和特殊token
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    #   -----------测试ppl--------------------
    if args.eval_ppl:
    # 默认使用 prompts 作为测试集
        test_texts = prompts
        ppl = compute_perplexity(model, tokenizer, test_texts)
        print(f"Perplexity: {ppl:.4f}")
        ppl1 = compute_perplexity(model, tokenizer, ["今天天气很好"])
        ppl2 = compute_perplexity(model, tokenizer, ["量子力学非常复杂"])
        ppl3 = compute_perplexity(model, tokenizer, ["测试"])
        ppl4 = compute_perplexity(model, tokenizer, ["你是谁？"])
        print(ppl1, ppl2, ppl3, ppl4)
    #   --------------------------------------


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
    计算因果语言模型在给定文本上的困惑度（Perplexity）。
    
    参数:
        model: MyLLMForCausalLM 实例（已加载权重，自动切换为 eval 模式）
        tokenizer: 对应分词器
        texts: str 或 List[str]，待评估的文本
        stride: 滑动窗口步长，默认为 max_length // 2
        max_length: 单次前向的最大 token 数，默认从 model.config.max_position_embeddings 读取
        device: 设备，默认使用 model 所在设备
    
    返回:
        float: 困惑度
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # 获取模型最大长度
    if max_length is None:
        # 根据你的 MyLLMConfig 属性名调整（可能是 max_seq_len）
        max_length = getattr(model.config, 'max_position_embeddings', 
                             getattr(model.config, 'max_seq_len', 512))
    if stride is None:
        stride = max_length // 2
    
    if isinstance(texts, str):
        texts = [texts]
    
    total_nll = 0.0      # 累积负对数似然（总损失）
    total_tokens = 0     # 累积有效 token 数量
    
    for text in tqdm(texts, desc="Computing Perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        for start in range(0, seq_len, stride):
            end = min(start + max_length, seq_len)
            input_window = input_ids[:, start:end]          # (1, L)

            # 正确构造标签：与输入对齐，模型内部会自动右移
            labels = input_window.clone()
            # 可选：最后一个位置标记为忽略（模型内部用不到，但无害）
            labels[:, -1] = -100

            with torch.no_grad():
                outputs = model(input_window, labels=labels, use_cache=False)
                loss = outputs.loss

            # 统计有效 token 数（标签非 -100 且被模型实际使用的部分）
            # 模型内部使用了 labels[:, 1:]，因此有效 token 数 = L-1
            num_valid = input_window.size(1) - 1
            if num_valid > 0:
                total_nll += loss.item() * num_valid
                total_tokens += num_valid
    
    if total_tokens == 0:
        return float('inf')
    
    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return perplexity

if __name__ == "__main__":
    main()