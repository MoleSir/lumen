from lumen import Tensor, DType
from lumen.nn import Parameter
from lumen.nn.modules import Module, Sigmoid, ModuleList
from typing import List


t = Tensor.randn((4, 4))
param = Parameter(Tensor.randn((4, 4)))

print(param)
print(t.requires_grad())

r = t + param

print(r.requires_grad())
gs = r.backward()

print(gs[param])
print(gs[t])

from lumen.nn.modules import Linear

l = Linear(10, 20, True)


print(l.get_parameter('weight').dims())
print(l.named_parameters())
for name, p in l.named_parameters():
    print(name, p.dims())
for name, p in l.named_buffers():
    print(name, p.dims())

print(l)
print(l.weight.dtype())
print(l.bias.dtype())
ll = l.to(dtype=DType.Float64)
print(l.weight.dtype())
print(l.bias.dtype())
print(l._parameters['weight'].dtype())


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
    

print(Mlp([2, 4, 1]))

import torch.nn.functional