import lumen
from lumen import Tensor
from lumen.nn.modules import Linear, Sigmoid, Module, MSELoss, ModuleList
from lumen.nn.optim import SGD
from typing import List


class Mlp(Module):
    def __init__(self, arch: List[int]):
        super().__init__()
        assert len(arch) >= 2

        linears = []
        activates = []
        for (in_dim, out_dim) in zip(arch, arch[1:]):
            linears.append(Linear(in_dim, out_dim, True))
            activates.append(Sigmoid())
        
        self.linears = ModuleList(linears)
        self.activates = ModuleList(activates)


    def layer_count(self):
        return len(self.linears)
    
    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.layer_count()):
            x = self.linears[i].forward(x)
            x = self.activates[i].forward(x)
        return x
    

def main():
    input = Tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])

    target = Tensor([[1.], [0.], [0.], [1.]])

    mlp = Mlp([2, 4, 1])
    optimizer = SGD(mlp.parameters(), 0.2)
    criterion = MSELoss()

    for n, p in mlp.named_parameters():
        print(n, p.dims())

    for i in range(10000):
        output = mlp.forward(input)
        loss = criterion.forward(output, target)
        grads = loss.backward()
        optimizer.step(grads)
    
    print(mlp.forward(input))


main()