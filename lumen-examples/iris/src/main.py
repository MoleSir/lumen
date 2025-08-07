import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载 Iris 数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target  # X: (150, 4), y: (150,)

# 2. 预处理数据
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 标准化数据，使均值为0，标准差为1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 3. 使用 DataLoader 处理数据
batch_size = 2
train_dataset = Data.TensorDataset(X_train, y_train)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 4. 定义神经网络
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 输入 4 维，隐藏层 10 维
        self.fc2 = nn.Linear(10, 3)  # 隐藏层 10 维，输出 3 类
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 初始化模型
model = IrisNet()
criterion = nn.MSELoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Adam 优化器

# 6. 训练模型
epochs = 100
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(batch_X)  # 前向传播
        batch_y = F.one_hot(batch_y, 3).float()
        loss = criterion(outputs, batch_y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if (epoch + 1) % 10 == 0:
        print(batch_X)
        print(batch_y)
        print(outputs)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 7. 评估模型
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)  # 取最大概率的类别
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
