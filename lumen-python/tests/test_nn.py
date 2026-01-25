from lumen import Tensor
from lumen.nn import Parameter

t = Tensor.randn((4, 4))
param = Parameter(Tensor.randn((4, 4)))

print(param)
print(t.requires_grad())

r = t + param

print(r.requires_grad())
gs = r.backward()

print(gs[param])
print(gs[t])

print('---', isinstance(param, Parameter))

from lumen.nn.modules import Linear

l = Linear(10, 20, True)

print(l.get_parameter('weight').dims())
print(l.named_parameters())
for name, p in l.named_parameters():
    print(name, p.dims())
for name, p in l.named_buffers():
    print(name, p.dims())


import torch

torch.nn.Sequential
torch.optim.SGD