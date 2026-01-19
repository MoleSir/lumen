import pytest
import lumen
from lumen import Tensor, DType


def test_tensor_init_list():
    t = Tensor([1, 2, 3])
    assert t.dims() == [3]


def test_tensor_init_nested():
    t = Tensor([[1., 2, 3], [1, 2, 3]])
    assert t.dims() == [2, 3]


def test_new_with_options():
    t = Tensor.new([[1, 2, 3], [4, 5, 6]], dtype=DType.Float64, requires_grad=True)
    assert t.dims() == [2, 3]
    assert t.dtype() == DType.Float64
    assert t.requires_grad() is True


def test_factory_methods():
    t1 = Tensor.zeros((3, 3, 3))
    assert t1.dims() == [3, 3, 3]

    t2 = Tensor.ones((5, 5), DType.Int32)
    assert t2.dtype() == DType.Int32


def test_arithmetic_ops():
    lhs = Tensor.ones((2, 3)) 
    rhs = Tensor.ones((2, 3)) 
    
    res = lhs + rhs
    assert res.allclose(Tensor.new([[2., 2., 2.], [2., 2., 2.]]))

    res_scalar = lhs + 2.0
    assert res_scalar.allclose(Tensor.new([[3., 3., 3.], [3., 3., 3.]]))


def test_slicing_basic():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # t[0] -> [1, 2, 3]
    assert t[0].dims() == [3]    
    # assert t[0, 1].item() == 2


def test_slicing_advanced():
    t = Tensor.randn((5, 5, 5))
    
    # t[0:3, 1:2, 1]
    slice_res = t[0:3, 1:2, 1]
    assert slice_res.dims() == [3, 1] 
    
    # lhs[[0, 1]]
    lhs = Tensor([[10, 20], [30, 40]])
    selected = lhs[[0, 1]]
    assert selected.dims() == []


def test_simple_backward():
    x = Tensor.ones((2, 2), requires_grad=True)
    y = Tensor.ones((2, 2), requires_grad=True)
    
    z = x + y # z = x + y, dz/dx = 1, dz/dy = 1
    
    assert z.requires_grad() is True
    
    grads = z.backward()
    
    assert x in grads
    assert y in grads
    
    expected = Tensor.ones((2, 2))
    assert grads[x].allclose(expected)
    assert grads[y].allclose(expected)


def test_complex_backward():
    x = Tensor([2.0], requires_grad=True)
    # y = x^2 * 3
    # dy/dx = 6x -> 当 x=2, grad=12
    y = (x * x) * 3.0 
    
    grads = y.backward()
    assert grads[x].item() == Tensor.new(12.0)
    

def test_no_grad_mode():
    with lumen.no_grad():
        x = Tensor.randn((2, 3), requires_grad=True)
        y = x + 1.0
        assert y.requires_grad() is False 
    
    z = Tensor.randn((2, 3), requires_grad=True)
    res = z + 1.0
    assert res.requires_grad() is True