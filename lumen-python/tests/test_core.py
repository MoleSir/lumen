import lumen
from lumen import Tensor, DType
import lumen.nn

lumen.nn.hello()

from tokenizers.models import Model


t1 = Tensor.zeros((3, 3, 3))
t2 = Tensor.ones((5, 5), DType.Int32)

print(t1.dims())
print(t1.dtype())

print(t2.dims())
print(t2.dtype())

print(Tensor.rand((2, 3)).dims())
print(Tensor.rand((2, 3), min=1.0, max=3.0, dtype=DType.Float64))

lhs = Tensor.randn((2, 3))
rhs = Tensor.ones((2, 3))
print(lhs.add(rhs))
print(lhs.add(2))
print(lhs.sub(rhs))
print(lhs.mul(rhs))
print(lhs.div(rhs))
print(Tensor.zeros((5, 3)).sin())

print(1 - lhs.relu())
print(3 * lhs.relu())

print('============')
print(lhs)
print('============')
print(lhs[0])
print('============')
print(lhs[(0)])
print('============')
print(lhs[(0, 2)])
print('============')
print(lhs[(1, 2)])
# print(lhs[[1, 2, 2]])
