from lumen.nn.modules import Module, Linear
from lumen.nn.optim import SGD
from lumen.dataset.datasets import MnistDataLoader, MnistDataset
import lumen.nn.functional as F
from lumen import Tensor
import lumen


class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 512, True)
        self.fc2 = Linear(512, 256, True)
        self.fc3 = Linear(256, 10, True)
    
    def forward(self, images: Tensor):
        (batch, height, width) = images.dims()
        x = images.reshape((batch, height * width))

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        probs = F.log_softmax(out, 1)

        return probs
    

def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 1

    train_dataset = MnistDataset('train', './cache')
    train_loader = MnistDataLoader(train_dataset, BATCH_SIZE, True)

    test_dataset = MnistDataset('test', './cache')
    test_loader = MnistDataLoader(test_dataset, BATCH_SIZE, True)

    model = Net()
    optimizer = SGD(model.parameters(), LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)


def train(model: Net, loader: MnistDataLoader, optimizer: SGD, epoch: int):
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
def test(model: Net, loader: MnistDataLoader):
    test_loss = 0

    for images, labels in loader:
        output = model(images)
        test_loss += F.nll_loss(output, labels).item()

    test_loss = test_loss / len(loader)

    print(f'Test set: Average loss: {test_loss}"')



if __name__ == '__main__':
    main()