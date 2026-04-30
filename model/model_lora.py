import torch
from torch import nn


# 定义LoRA网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank    # LoRA的秩
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全零初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))
    

def apply_lora(model, rank=16):
    for name, module in model.named_modules():  # 遍历模型的所有模块，找到需要添加lora的线性层
        if isinstance(module, nn.Module) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, 'lora', lora)   # 用setattr将lora作为模块的一个属性
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 闭包延迟绑定：在循环中定义函数，如果函数体内引用循环变量（如 original_forward, lora），这些变量在函数调用时才求值，由于循环变量不断变化，所有函数最终都会使用最后一次迭代的值。
                # 默认参数的值在函数定义时就求值并绑定，所以即使循环变量变了，默认参数还保留当时的值。
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora


def load_lora(model, path):
    """加载lora权重文件"""
    # 如果权重来自 DataParallel 或 DistributedDataParallel 保存（键名以 module. 开头），则去掉该前缀，保证与当前模型键名对齐
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """保存lora权重文件"""
    # 若模型被 torch.compile 处理过，model 对象会有 _orig_mod 属性指向原始模型，否则就直接用 model
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 键名中若以 module. 开头则去掉
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    # 将 LoRA 增量合并到原始权重并保存
    load_lora(model, lora_path)     # 训练好的LoRA权重载入模型
    raw_model = getattr(model, '_orig_mod', model)      # 获取原始模型

    # 初始化一个 state_dict，从原始模型状态字典中复制所有非 lora 参数（即排除含 .lora. 的键），并转为 FP16/CPU
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}    
    # 对所有 nn.Linear 模块（且键名不含 .lora. 避免重复处理）
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 重新用 .weight.data 覆盖 state_dict 中对应权重的值
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 若该模块有 lora 属性，则将合并的增量 B @ A 加到该权重上
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
    torch.save(state_dict, save_path)