from lumen.dataset.datasets import MnistDataset, MnistDataLoader
from lumen import Tensor

d = MnistDataset('test', './cache')

loader = MnistDataLoader(d, 2, True)

zeros = Tensor.zeros((2, 28, 28))
for i, (image, label) in enumerate(loader):
    print(zeros.allclose(image))
    print(label)
