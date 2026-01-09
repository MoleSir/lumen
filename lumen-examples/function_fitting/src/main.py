import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import math
import matplotlib.pyplot as plt

class FunctionDataSet(Dataset):
    MIN_FUNC_X = 0.
    MAX_FUNC_X = 10
    MIN_FUNC_SEQ_LEN = 10
    MAX_FUNC_SEQ_LEN = 100

    def __init__(self, func, num_samples):
        """
            - `func`: 一个接受 float 类型输入，返回 float 类型输出的函数
            - `num_samples`: 数据集中样本的数量
            - `seq_len`: 每次获取数据得到多少组输入输出映射
        """
        self.func = np.vectorize(func)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, _):
        seq_len = random.randint(self.MIN_FUNC_SEQ_LEN, self.MAX_FUNC_SEQ_LEN)
        xs = np.linspace(self.MIN_FUNC_X, self.MAX_FUNC_X, seq_len)
        ys = self.func(xs)
        
        x_tensor = torch.tensor(xs, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)

        return x_tensor, y_tensor


class Model(nn.Module):
    def __init__(self, hidden_size: int, rnn: str='rnn'):
        super(Model, self).__init__()
        self.input_size = 1
        self.hidden_size = hidden_size
        self.output_size = 1
        
        if rnn == 'rnn':
            self.model = nn.RNN
        elif rnn == 'lstm':
            self.model = nn.LSTM
        elif rnn == 'gru':
            self.model = nn.GRU
        self.rnn = self.model(self.input_size, self.hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, self.output_size)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, 1)
        """
        seq_len = x.size(1)
        # r_outs: (batch_size, seq_len, hidden_size)
        r_outs, _ = self.rnn(x)
        # 需要处理 r_outs 中的每个 seq
        outs = [ self.step(r_outs[:, time, :]) for time in range(seq_len) ]
        # 输出：(batch_size, seq_len, 1)
        return torch.stack(outs, dim=1)

    def step(self, r_out):
        """
            r_out: each seq after rnn (batch_size, hidden_size)
        """
        # out: (batch_size, 2*hidden_size)
        out = torch.tanh(self.fc1(r_out))
        # out: (batch_size, 1)
        out = self.fc2(out)
        return out


def get_dataloader(func, train_samples, test_sample):
    train_dataset = FunctionDataSet(func, train_samples)
    train_dataloader = DataLoader(train_dataset, 1)

    test_loader = FunctionDataSet(func, test_sample)
    test_dataloader = DataLoader(test_loader, 1)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    print('function fitting~')
    TRAIN_SMAPLES = 5000
    TEST_SAMPLES = 1

    def func(x: float) -> float:
        y = math.cos(2 * x)
        return y

    train_dataloader, test_dataloader = get_dataloader(func, TRAIN_SMAPLES, TEST_SAMPLES)
    model = Model(5, 'gru')
    cirterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    model.train()
    for i, (x, y) in enumerate(train_dataloader):
        pred = model(x)
        loss = cirterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        if i % 100 == 0:
            print('loss:', loss.item())
        optimizer.step()

    for x, y in test_dataloader:
        pred = model(x)
        x = x.detach().numpy().flatten()
        pred = pred.squeeze().detach().numpy().flatten()
        plt.plot(x, pred, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.title("Line Plot with Grid and Axis Range")
        plt.xlabel("x values")
        plt.ylabel("y values")
        plt.show()