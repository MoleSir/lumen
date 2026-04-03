"""
    Reference: https://github.com/rossning92/ai-lyrics-writing
"""

import datetime
import glob
import os
import sys
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

DEBUG = False
EMBED_SIZE = 128
HIDDEN_SIZE = 1024
LEARN_RATE = 0.001
LSTM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 15
SEQ_LEN = 48

if DEBUG:
    BATCH_SIZE = 2
    EPOCHS = 1000


class LyricsDataset(Dataset):
    def __init__(self, seq_len: int, file: str='./data/lyrics.txt'):
        SOS = 0 # Start of song
        EOS = 1 # End of song

        self.seq_len = seq_len
        with open(file, encoding='utf-8') as f:
            lines = f.read().splitlines()

        self.word2index = { '<SOS>': SOS, '<EOS>': EOS }

        # Convert words to indices
        # `indices`: map all word in file to index(also add `SOS` and `EOS` to divide song)
        indices = []
        num_words = 0
        for line in lines:
            # For each line
            indices.append(SOS)
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words
                    num_words += 1
                indices.append(self.word2index[word])
            indices.append(EOS)

        self.index2word = { v: k for v, k in self.word2index.items() }
        self.data = np.array(indices, dtype=np.int64)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
    
    def __getitem__(self, index):
        start = index * self.seq_len
        end = start + self.seq_len
        # 取出 seq_len 个词作为一个 Batch
        # 输出是想对输入后移一个词，预测下一个字

        # 输出：
        # input:  (seq_len) => 其中每个数字代表一个词
        # output: (seq_len) => 其中每个数字代表一个词
        return (
            torch.as_tensor(self.data[start:end]), # input 
            torch.as_tensor(self.data[start+1:end+1]), # output
        )
    

class LyricsNet(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, lstm_layers: int):
        super(LyricsNet, self).__init__()
        # Embedding 将使用标量代表的词转为一个向量
        # vocab_size 表示词语的数量（那么代表词的标量范围为 [0, vocab_size)）
        # embed_size 表示经过 Embedding，每个词会被转为一个 embed_size 长度的向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # lstm 的输入：(batch_size, seq_len, embed_size)
        # lstm 的输出：(batch_size, seq_len, hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, lstm_layers, batch_first=True)
        # h2h 的输入：(batch_size, seq_len, hidden_size)
        # h2h 的输出：(batch_size, seq_len, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # h2o 的输入：(batch_size, seq_len, hidden_size)
        # h2o 的输出：(batch_size, seq_len, vocab_size) 
        # => 表示若干个 batch，其中每个 batch 有 seq_len 的词，每个词用 vocab_size 维度的向量表示
        self.h2o = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_ids, lstm_hidden=None):
        """
            输入 word_ids: (batch_size, seq_len)
            输出： (batch_size, seq_len, embed_size)
        """
        # embedded: (batch_size, seq_len, embed_size)
        embedded = self.embedding(word_ids)

        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        out = F.tanh(self.h2h(lstm_out))
        out = self.h2o(out)

        return out, lstm_hidden
    

def accuracy(output, target):
    """
    Args:
        output: 由模型输出：(batch_size, seq_len, vocab_size) 
        target: 有 Dataloader 输出：(batch_size, seq_len) 

    Returns:
        float: accuracy value between 0 and 1 
    """
    vocab_size = output.size(2)
    # output: (batch_size * seq_len, vocab_size)
    output: torch.Tensor = output.reshape(-1, vocab_size)

    # 返回张量中前 k 个最大值及其索引
    # output 中的 (N * vocab_size) => vocab_size 是向量，代表一个词
    # 这里就是 N 个词，N = batch_size * seq_len 和 target 大小一致（target 中一个 item 就代表一个词！）
    a = output.topk(1).indices.flatten()
    b = target.flatten()
    return a.eq(b).sum().item() / len(a)

def generate(start_phrases):
    # i.e. '宁可/无法' => ['宁可', '无法']
    start_phrases = start_phrases.split("/")

    hidden = None

    def next_word(input_word):
        nonlocal hidden

        input_word_index = dataset.word2index[input_word]
        input_ = torch.Tensor([[input_word_index]]).long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return dataset.index2word[top_word_index]
    
    result = []
    cur_word = '/'

    for i in range(SEQ_LEN):
        if cur_word ==  '/':
            result.append(cur_word)
            next_word(cur_word)

            if len(start_phrases) == 0:
                break

            for w in start_phrases.pop(0):
                result.append(w)
                cur_word = next_word(w)
        else:
            result.append(cur_word)
            cur_word = next_word(cur_word)
        
    result = "".join(result)
    result = result.strip("/")  # remove trailing slashes
    return result


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = LyricsDataset(seq_len=SEQ_LEN)

    data_length = len(dataset)
    lengths = [int(data_length - 1000), 1000]
    # 取出 1000 个作为测试集
    train_data, test_data = random_split(dataset, lengths)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    if DEBUG:
        train_loader = [next(iter(train_loader))]
        test_loader = [next(iter(test_loader))]

    # vocab_size 表示我们使用标量表示词的数量，word2index 中索引的范围就是 [0, vocab_size)
    # 每个 item 代表一个词
    vocab_size = len(dataset.word2index)

    model = LyricsNet(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    cirterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for i, (input, target) in enumerate(train_loader):
            model.train()

            # input:  (batch_size, seq_len)
            # target: (batch_size, seq_len)
            input, target = input.to(device), target.to(device)

            # output: (batch_size, seq_len, embed_size)
            output, _ = model(input)
            loss = cirterion(output.reshape(-1, vocab_size), target.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(output, target)
            print(
                "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
                % (epoch, i, len(train_loader), loss.item(), acc)
            )            

