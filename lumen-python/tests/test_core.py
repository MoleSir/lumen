import lumen
from lumen import Tensor, DType
import lumen.nn


def test_new():
    t = Tensor([1, 2, 3])
    print(t)

    t = Tensor([[1., 2, 3], [1, 2, 3]])
    print(t)

    t = Tensor.new([[1., 2, 3], [4, 5, 6]], dtype=DType.Float64)
    print(t)

    t = Tensor.new([[1, 2, 3], [4, 5, 6]], dtype=DType.Float64, requires_grad=True)
    print(t, t.requires_grad())


def test_base_op():
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
    print(lhs[(1)])
    print('============')
    print(lhs[(0, 2)])
    print('============')
    print(lhs[(1, 2)])
    print('============')
    print(lhs[[0, 1]])
    # print(lhs[[1, 2, 2]])

    t = Tensor.randn((5, 5, 5))

    print(t[0:3, 1:2, 1])


def test_base_grad():
    lhs = Tensor.randn((2, 3))
    rhs = Tensor.ones((2, 3))
    print(lhs.requires_grad(), rhs.requires_grad())

    lhs.set_requires_grad(True)
    rhs.set_requires_grad(True)
    print(lhs.requires_grad(), rhs.requires_grad())

    res = lhs + rhs
    print(res.requires_grad())

    grads = res.backward()

    print(grads[lhs])
    print(grads[rhs])

    print(grads[lhs].allclose(grads[rhs]))

    for id, tensor in grads.items():
        print(id, tensor)


def test_no_grad():
    with lumen.no_grad():
        lhs = Tensor.randn((2, 3))
        rhs = Tensor.ones((2, 3))

        lhs.set_requires_grad(True)
        rhs.set_requires_grad(True)

        res = lhs + rhs
        print(res.requires_grad())

    lhs = Tensor.randn((2, 3))
    rhs = Tensor.ones((3))

    lhs.set_requires_grad(True)
    rhs.set_requires_grad(True)

    res = lhs + rhs
    print(res.requires_grad())


def test_shape():
    pass


def test_condition():
    pass


test_new()
# test_no_grad()
# test_base_grad()