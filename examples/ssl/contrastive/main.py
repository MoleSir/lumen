import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cpu")

base_dataset = datasets.MNIST(root="../cache", train=True, download=True)

transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

class ContrastiveDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]   # img 是 PIL
        x1 = self.transform(img)    # → Tensor
        x2 = self.transform(img)    # → Tensor
        return x1, x2

    def __len__(self):
        return len(self.dataset)

dataset = ContrastiveDataset(base_dataset, transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 2. Encoder + Projection Head
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        h = self.net(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

model = Encoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

"""
所以 z1 是 N 张图片，z2 也是 N 张图片，我们应该让 z1_1（z1 的第一张图） 接近 z2_1（z2 的第一张图，他们都是由同一个图片经过不同噪声得到的），原理 z2_i（i = 2 .. N）。
而这个要求是对所有的 i 都要满足：z1_i 要接近 z2_i，远离其他的 z2_j。
positives = torch.cat([
torch.diag(sim, N),   # 取右上角对角线 (z1 vs z2)
torch.diag(sim, -N)   # 取左下角对角线 (z2 vs z1)
])
这个部分torch.diag(sim, N)：得到了所有 z1_x 和 z2_x 的 dot
torch.diag(sim, -N)：得到了所有 z2_x 和 z1_x 的 dot
要求这些值要大，dot 越大，向量越靠近（实际上这个值是重复的吧，dot 操作可以交换左右向量）

然后mask = torch.eye(2*N, dtype=torch.bool)
sim.masked_fill_(mask, -9e15) # 排除自己
denominator = torch.logsumexp(sim, dim=1)
这个部分计算与其他图片的差异。
ltorch.logsumexp(sim, dim=1) 就是遍历 sim(2N x 2N)的第一个索引：i in 0-2N，计算 sim[i] 这个 2N 向量的每个元素的指数和然后取 log。而这 2N 的向量刚好是 x1_1 对所有 2N 向量的 dot，排除 x1_1 对本身，就得到了 x1_1 对其他所有向量的 sim 值，这个值要求越小越好！
"""
def nt_xent(z1, z2, temperature=0.5):
    N = z1.size(0)
    """
    [z1_1
    z1_2
    ...
    z1_N
    z2_1
    z2_2
    ...
    z2_N]
    """
    z = torch.cat([z1, z2], dim=0)

    # [2N, D] @ [D, 2N]，相当于上面的向量互相 dot 一次，得到 [2D, 2D]
    sim = torch.matmul(z, z.T) / temperature
    # 去掉自己和自己的比较
    mask = torch.eye(2*N, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)

    """
        (0, N) => z1_1 dot z2_1
        (1, N+1) => z1_2 dot z2_2
        ...
        (N-1, 2N-1)

        和 

        (N, 0) => z2_1 dot z1_1
        (N+1, 1) => z2_2 dot z1_2
        ...

    """
    positives = torch.cat([
        torch.diag(sim, N),
        torch.diag(sim, -N)
    ])

    denominator = torch.logsumexp(sim, dim=1)
    loss = -positives + denominator
    return loss.mean()

# 3. 训练
for epoch in range(3):
    total_loss = 0
    for x1, x2 in loader:        
        z1 = model(x1)
        z2 = model(x2)

        loss = nt_xent(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")