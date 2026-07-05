import torch
import torch.nn as nn
import math

class BertEmbeddings(nn.Module):
    """包含词嵌入、位置嵌入和段落嵌入"""
    def __init__(self, vocab_size, hidden_size, max_len, token_type_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.token_type_embeddings = nn.Embedding(token_type_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0).expand_as(input_ids)
        
        # 融合三种 Embedding
        emb = (self.token_embeddings(input_ids) + 
               self.position_embeddings(pos) + 
               self.token_type_embeddings(token_type_ids))
        return self.dropout(self.layer_norm(emb))

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        # 1. 线性变换并分头
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 计算注意力分数 (Scaled Dot-Product)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v) # [B, H, L, D]
        
        # 3. 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out(context)

class BertLayer(nn.Module):
    """一个标准的 BERT 层（Transformer Encoder Block）"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(self.attention(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class BERT(nn.Module):
    """完整的 BERT 模型结构"""
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, max_len=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_len, token_type_size=2)
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, hidden_size * 4) 
            for _ in range(num_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size) # 用于取出 [CLS] 向量

    def forward(self, input_ids, token_type_ids, mask=None):
        # mask shape: [batch_size, 1, 1, seq_len] 用于广播
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, mask)
        
        # 取出第一个 token [CLS] 的输出作为句向量表示
        cls_out = torch.tanh(self.pooler(x[:, 0]))
        return x, cls_out

# 实例化测试
model = BERT(vocab_size=30000)
sample_ids = torch.randint(0, 30000, (2, 10)) # batch_size=2, seq_len=10
sample_segments = torch.zeros(2, 10, dtype=torch.long)
sequence_output, pooled_output = model(sample_ids, sample_segments)
print(f"序列输出形状: {sequence_output.shape}") # [2, 10, 768]
print(f"池化输出形状: {pooled_output.shape}")   # [2, 768]