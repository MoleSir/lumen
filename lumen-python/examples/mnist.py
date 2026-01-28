from lumen.nn.modules import Module, Linear, Dropout, Relu
from lumen.nn.optim import SGD
from lumen.dataset.datasets import MnistDataLoader, MnistDataset
import lumen.nn.functional as F
from lumen import Tensor
import lumen


class MnistNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 512, True)
        self.fc2 = Linear(512, 256, True)
        self.fc3 = Linear(256, 10, True)
        self.dropout = Dropout(0.2)
        self.relu = Relu()
    
    def forward(self, images: Tensor):
        (batch, height, width) = images.dims()
        x = images.reshape((batch, height * width))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        probs = F.log_softmax(x, 1)

        return probs

    @classmethod
    def init(cls, _config, init = None):
        return MnistNet()


def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 1

    train_dataset = MnistDataset('train', './cache')
    train_loader = MnistDataLoader(train_dataset, BATCH_SIZE, True)

    test_dataset = MnistDataset('test', './cache')
    test_loader = MnistDataLoader(test_dataset, 1000, True)

    model = MnistNet()
    optimizer = SGD(model.parameters(), LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)

    model.save_safetensors('./cache/mnist.model.safetensors')

    model = MnistNet.from_safetensors(None, './cache/mnist.model.safetensors')
    test(model, test_loader)


def train(model: MnistNet, loader: MnistDataLoader, optimizer: SGD, epoch: int):
    for batch_index, (images, labels) in enumerate(loader): 
        output = model(images)
        loss = F.nll_loss(output, labels)
        grads = loss.backward()
        optimizer.step(grads)

        if batch_index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {}'.format(
                epoch,
                batch_index * loader.batch_size,
                loader.dataset_len(),
                100 * batch_index / len(loader),
                loss.item()
            ))


@lumen.no_grad()
def test(model: MnistNet, loader: MnistDataLoader):
    model.eval()
    test_loss = 0
    correct = 0

    for images, labels in loader:
        output: Tensor = model(images)
        test_loss += F.nll_loss(output, labels).item()
        correct += output.argmax(1, keep_dim=True).eq(labels).true_count()
        
    test_loss = test_loss / len(loader)
    accuracy = 100 * correct / loader.dataset_len()

    print(f'Test set: Average loss: {test_loss}", Accuracy: {correct}/{loader.dataset_len()} ({accuracy})')



if __name__ == '__main__':
    main()