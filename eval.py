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
        # 用独立的变量，避免覆盖全局 prompts
        ppl_test_prompts = ['量子力学好难', '你是谁', '今天天气真好', '解释什么是机器学习']
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
    import torch
    from tqdm import tqdm

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if max_length is None:
        max_length = getattr(model.config, 'max_position_embeddings',
                             getattr(model.config, 'max_seq_len', 512))
    if stride is None:
        stride = max_length // 2

    if isinstance(texts, str):
        texts = [texts]

    total_nll = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Computing Perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        # 第一个窗口从 0 开始
        start = 0
        while start < seq_len:
            end = min(start + max_length, seq_len)
            input_window = input_ids[:, start:end]          # (1, L)
            labels = input_window.clone()

            # 只让窗口的“最后 stride 个 token”参与损失计算（第一个窗口全算）
            # 第一个窗口：允许所有位置计算（即不额外屏蔽）
            if start != 0:
                # 屏蔽掉前面的重叠部分，只保留最后 stride 个 token 的 label
                labels[:, :-stride] = -100

            # 注意：不要手动屏蔽最后一个 token！模型内部 shift 已处理
            with torch.no_grad():
                outputs = model(input_window, labels=labels, use_cache=False)
                loss = outputs.loss   # 这已经是只对 labels 中非 -100 位置的平均损失

            # 统计当前窗口实际参与计算的有效 token 数（等于 labels 中非 -100 的数量）
            num_valid = (labels[:, 1:] != -100).sum().item()
            # 对于最后一个窗口，如果 end == seq_len，有效 token 数可能需要调整？
            # 模型内部 shift 会忽略 labels 的第一个有效 token（因为 logits 少一位），
            # 但 CrossEntropyLoss 已经自动处理了 -100，所以直接按 labels 的非 -100 计数即可。
            if num_valid > 0:
                total_nll += loss.item() * num_valid
                total_tokens += num_valid

            start += stride

    if total_tokens == 0:
        return float('inf')

    avg_nll = total_nll / total_tokens
    return torch.exp(torch.tensor(avg_nll)).item()

if __name__ == "__main__":
    main()