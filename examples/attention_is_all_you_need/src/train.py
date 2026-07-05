class NoamOpt:
    """原论文中的学习率调度策略 (Warmup + Inverse Square Root Decay)"""
    def __init__(self, d_model, warmup_steps, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        # 论文中的公式: d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        return (self.d_model ** (-0.5)) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
    
    def zero_grad(self):
        self.optimizer.zero_grad()


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer

# 使用多语言 BPE Tokenizer 代替从头训练词表
# 这包含了原论文所需的特殊符号 <pad>, <s>, </s> (即 SOS, EOS)
TOKENIZER_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

PAD_IDX = tokenizer.pad_token_id  # 获取真实的 PAD 对应的 ID
SOS_IDX = tokenizer.cls_token_id  # 使用 CLS 作为句首标记 <SOS>
EOS_IDX = tokenizer.sep_token_id  # 使用 SEP 作为句尾标记 <EOS>
VOCAB_SIZE = tokenizer.vocab_size

class WMT14Dataset(Dataset):
    def __init__(self, split="train", max_samples=100000, max_len=128):
        super().__init__()
        # 加载真实的 WMT14 英德翻译数据集 (原论文用的就是这个)
        # 注意：完整数据集有450万条，为了你能在本地跑起来，这里截取前 max_samples 条
        print(f"正在加载真实的 WMT14 数据集 ({split})...")
        raw_data = load_dataset("wmt14", "de-en", split=f"{split}[:{max_samples}]")
        
        self.data = []
        self.max_len = max_len
        
        print("正在进行 BPE 分词处理...")
        for item in raw_data:
            en_text = item['translation']['en']
            de_text = item['translation']['de']
            
            # 使用 Tokenizer 将真实文本转为 ID
            src_ids = tokenizer.encode(en_text, add_special_tokens=False)
            tgt_ids = tokenizer.encode(de_text, add_special_tokens=False)
            
            # 过滤掉过长的句子 (原论文按长度做过筛选)
            if len(src_ids) <= max_len - 2 and len(tgt_ids) <= max_len - 2:
                self.data.append((src_ids, tgt_ids))
                
        print(f"数据处理完毕，共保留 {len(self.data)} 条高质量句子对。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # 严格按照规范：源句子不需要加 SOS/EOS，或者只加 EOS；
        # 目标句子必须是: <SOS> token1 token2 ... <EOS>
        src_tensor = torch.tensor(src)
        tgt_tensor = torch.tensor([SOS_IDX] + tgt + [EOS_IDX])
        
        return src_tensor, tgt_tensor

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # 用真实的 PAD_IDX 进行对齐
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_padded, tgt_padded