import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cpu")

# 1. 数据
transform = transforms.ToTensor()
dataset = datasets.MNIST(root="../cache", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 2. 简单自编码器
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(-1, 1, 28, 28)

model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 3. 训练
for epoch in range(3):
    total_loss = 0
    for x, _ in loader:
        x = x.to(device)

        # 加噪声
        noise = torch.randn_like(x) * 0.3
        x_noisy = torch.clamp(x + noise, 0., 1.)

        x_hat = model(x_noisy)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")