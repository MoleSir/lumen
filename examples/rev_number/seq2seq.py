import torch
import torch.nn as nn
import torch.optim as optim
import random

# --- 1. 配置参数 ---
# 词表大小：数字 0-9，加上 SOS(开始标志)=10, EOS(结束标志)=11
VOCAB_SIZE = 12 
SOS_TOKEN = 10
EOS_TOKEN = 11

# 模型参数 (设得很小，CPU 随便跑)
EMBED_DIM = 16
HIDDEN_DIM = 32
SEQ_LEN = 6      # 序列长度
LR = 0.01
EPOCHS = 2000    # 迭代次数

# --- 2. 定义 Seq2Seq 模型 ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        # 把数字索引转换成向量
        self.embedding = nn.Embedding(input_dim, emb_dim)
        # LSTM 处理序列
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src: [batch_size, seq_len]
        embedded = self.embedding(src) # -> [batch, seq_len, emb_dim]
        
        # LSTM 返回: output, (hidden, cell)
        # 我们只需要 hidden 和 cell (这就是 Context Vector)
        _, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        # 最后把 hidden state 转换成词表大小的概率分布
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        # input_token: [batch_size, 1] (当前这一步输入的词)
        # hidden, cell: 来自 Encoder 或 上一步 Decoder 的状态
        
        # 1. 增加维度: [batch, 1] -> [batch, 1, emb_dim]
        embedded = self.embedding(input_token)
        
        # 2. LSTM 跑一步
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # 3. 预测下一个词: [batch, 1, hidden] -> [batch, 1, vocab_size]
        prediction = self.fc_out(output)
        
        # 去掉序列维度: [batch, vocab_size]
        return prediction.squeeze(1), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: 输入序列 [batch, seq_len]
        # trg: 目标序列 [batch, seq_len] (用来训练)
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = VOCAB_SIZE
        
        # 准备一个容器存储 Decoder 每一步的输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        
        # 1. 【Encoder 阶段】：把 source 跑完，拿到 context vector
        hidden, cell = self.encoder(src)
        
        # 2. 【Decoder 阶段】
        # Decoder 的第一个输入是 <SOS> (Start Of Sentence)
        input_token = trg[:, 0].unsqueeze(1) # 取出 SOS
        
        for t in range(1, trg_len):
            # 跑一步 Decoder
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            
            # 存下预测结果
            outputs[:, t, :] = output
            
            # Teacher Forcing: 训练时，下一步的输入直接用正确答案
            # (也可以用 output.argmax(1) 自己预测的，但刚开始训练用正确答案收敛快)
            input_token = trg[:, t].unsqueeze(1)
            
        return outputs

# --- 3. 数据生成器 ---
def get_data():
    # 生成随机序列: [1, 5, 2, 9]
    seq = [random.randint(0, 9) for _ in range(SEQ_LEN)]
    
    # 输入: [1, 5, 2, 9]
    src = torch.LongTensor([seq]) # 加个 batch 维度
    
    # 目标: SOS + [9, 2, 5, 1] + EOS
    target_seq = [SOS_TOKEN] + seq[::-1] + [EOS_TOKEN]
    trg = torch.LongTensor([target_seq])
    
    return src, trg

# --- 4. 训练 ---
# 初始化模型
enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
model = Seq2Seq(enc, dec)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("🚀 开始训练 Seq2Seq (CPU版)...")

for epoch in range(EPOCHS):
    model.train()
    src, trg = get_data()
    
    optimizer.zero_grad()
    
    # 前向传播
    output = model(src, trg)
    
    # 计算 Loss
    # output: [batch, trg_len, vocab_size] -> reshape -> [batch*trg_len, vocab_size]
    # trg: [batch, trg_len] -> reshape -> [batch*trg_len]
    # 忽略第0个(SOS)，从第1个开始算 loss
    loss = criterion(output[:, 1:].reshape(-1, VOCAB_SIZE), trg[:, 1:].reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\n✅ 训练完成！开始测试...")

# --- 5. 测试 (Inference) ---
model.eval()
with torch.no_grad():
    # 生成新数据
    src, trg = get_data()
    print(f"输入序列: {src.squeeze().tolist()}")
    
    # 手动跑一遍 Inference (不使用 Teacher Forcing)
    # 1. Encode
    hidden, cell = model.encoder(src)
    
    # 2. Decode
    input_token = torch.LongTensor([[SOS_TOKEN]]) # 初始输入 SOS
    pred_seq = []
    
    for _ in range(SEQ_LEN + 1): # 最多预测这么长
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        
        # 选概率最大的词
        pred_token = output.argmax(1).item()
        
        if pred_token == EOS_TOKEN:
            break
            
        pred_seq.append(pred_token)
        
        # 把预测的词作为下一步输入
        input_token = torch.LongTensor([[pred_token]])
        
    print(f"模型预测: {pred_seq}")
    print(f"正确答案: {trg.squeeze().tolist()[1:-1]}") # 去掉SOS和EOS看中间