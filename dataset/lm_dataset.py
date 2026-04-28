from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#os.environ["HF_DATASETS_OFFLINE"] = "1"
#os.environ["HF_HUB_OFFLINE"] = "1"


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是myLLM，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是myLLM，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are myLLM, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are myLLM, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    # 需要实现__len__和__getitem__
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 文本转化成input_id
        # max_length 预留bos，eos位置，若长度超过则truncation
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        # 拼接bos，eos
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # padding
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # 没有attention_mask 部分？
        return input_ids, labels
    

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({
            'conversations': [{
                'role': Value('string'), 
                'content': Value('string'), 
                'reasoning_content': Value('string'), 
                'tools': Value('string'), 
                'tool_calls': Value('string')
                }]
            })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        # self.bos_id: [1(bos), xxx(assistant), yyy(\n)]
        # self.eos_id: [2(eos), yyy(\n)]

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # return形如<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )
        
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)    # 因为-100不参与交叉熵运算
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        # 形如<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|> 
        # ---->
        # return ..., x(Hi), 2(eos), ..  其他位置全为-100
        return labels
    
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]  # 防止超过最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))     # 保持批次内长度一致，不足max_length的部分用padding补足
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
