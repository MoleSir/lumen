import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        print(Q.shape)
        print(K.shape)
        print(mask.shape)
        print('==============')

        # d_k 是特征维度
        d_k = Q.size(-1)
        # 1. 计算得分: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. 如果有 mask，将对应位置设为极小值，这样 softmax 后权重接近 0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Softmax 归一化权重
        attn = F.softmax(scores, dim=-1)
        
        # 4. 权重乘以 V
        context = torch.matmul(attn, V)
        return context, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 定义 W_q, W_k, W_v 矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换后切分为多头: (batch, n_heads, seq_len, d_k)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # 计算注意力
        context, attn = self.attention(q_s, k_s, v_s, mask)
        
        # 拼接多头并还原维度
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(context)
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.fc(x)
    


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # 1. Multi-head Attention + Residual + LayerNorm
        attn_out = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_out)
        # 2. FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, n_heads) # Masked MHA
        self.mha2 = MultiHeadAttention(d_model, n_heads) # Encoder-Decoder MHA
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, dec_inputs, enc_outputs, self_mask, cross_mask):
        # 1. Masked Self-Attention
        out = self.ln1(dec_inputs + self.mha1(dec_inputs, dec_inputs, dec_inputs, self_mask))
        # 2. Encoder-Decoder Attention (Q来自解码器，K, V来自编码器)
        out = self.ln2(out + self.mha2(out, enc_outputs, enc_outputs, cross_mask))
        # 3. FFN
        out = self.ln3(out + self.ffn(out))
        return out
    


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def get_pad_mask(self, x, pad_idx=0):
        # 用于遮盖 Padding 符号
        return (x != pad_idx).unsqueeze(-2)

    def get_subsequent_mask(self, x):
        # 用于解码器的掩码，防止看到未来的信息
        batch_size, seq_len = x.size()
        subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        return ~subsequent_mask

    def forward(self, src, tgt):
        # 1. 处理 Mask
        src_mask = self.get_pad_mask(src)
        tgt_mask = self.get_pad_mask(tgt) & self.get_subsequent_mask(tgt)
        
        # 2. Encoder
        enc_out = self.pos_emb(self.src_emb(src))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
            
        # 3. Decoder
        dec_out = self.pos_emb(self.tgt_emb(tgt))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
            
        # 4. Final Projection
        logits = self.projection(dec_out)
        return logits
    


if __name__ == '__main__':
    # 参数设置
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    model = Transformer(src_vocab_size, tgt_vocab_size)

    src = torch.LongTensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]])
    tgt = torch.LongTensor([[1, 2, 3, 4, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0]])

    output = model(src, tgt)
    print(output.shape)

