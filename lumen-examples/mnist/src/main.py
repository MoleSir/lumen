import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 设置超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.01  # MLP 通常可以使用稍大一点的学习率
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {DEVICE}")

# 2. 数据准备 (和之前一样)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# 3. 定义全连接网络模型 (MLP)
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 输入层 -> 隐藏层1
        # 输入维度: 28*28 = 784 (MNIST图片像素总数)
        # 输出维度: 512 (这是一个超参数，可以自己定，通常选2的倍数)
        self.fc1 = nn.Linear(784, 512)
        
        # 隐藏层1 -> 隐藏层2
        self.fc2 = nn.Linear(512, 256)
        
        # 隐藏层2 -> 输出层
        # 输出维度: 10 (对应 0-9 十个数字)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # 关键步骤：展平 (Flatten)
        # 输入 x 的形状是 [batch_size, 1, 28, 28]
        # 我们需要把它变成 [batch_size, 784]
        # -1 表示自动计算 batch_size 维度，28*28 是特征维度
        x = x.view(-1, 28 * 28) 
        
        # 第一层 + 激活函数 (ReLU)
        x = self.fc1(x)
        x = F.relu(x)
        
        # 第二层 + 激活函数
        x = self.fc2(x)
        x = F.relu(x)
        
        # 输出层
        x = self.fc3(x)
        
        # Log Softmax 用于计算概率 (配合 NLLLoss)
        return F.log_softmax(x, dim=1)

# 初始化模型
model = MLPNet().to(DEVICE)

# 4. 优化器
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# 这里换成了 SGD，对于简单的 MLP 有时候 SGD 效果更稳，当然用 Adam 也可以

# 5. 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 6. 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# 7. 运行
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)