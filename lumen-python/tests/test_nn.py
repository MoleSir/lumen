import pytest
from typing import List
from lumen import Tensor, DType
from lumen.nn import Parameter
from lumen.nn.modules import Module, Sigmoid, ModuleList, Linear


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


def test_autograd_basic():
    shape = (4, 4)
    t = Tensor.randn(shape)
    param = Parameter(Tensor.randn(shape))

    assert param.requires_grad() is True
    
    r = t + param
    assert r.requires_grad() is True
    assert r.dims() == list(shape)

    gs = r.backward()

    assert param in gs
    assert t in gs
    
    grad_param = gs[param]
    grad_t = gs[t]
    
    assert grad_param.dims() == list(shape)
    assert grad_t.dims() == list(shape)


def test_linear_module_structure():
    in_features = 10
    out_features = 20
    l = Linear(in_features, out_features, bias=True)

    assert l.parameter_count() == 2 
    assert l.parameter_element_count() == (in_features * out_features) + out_features

    weight = l.get_parameter('weight')
    bias = l.get_parameter('bias')
    
    assert weight is not None
    assert bias is not None
    
    params_dict = dict(l.named_parameters())
    assert 'weight' in params_dict
    assert 'bias' in params_dict
    assert params_dict['weight'].dims() == weight.dims()


def test_module_dtype_cast():
    l = Linear(10, 20, True)

    l = l.to(dtype=DType.Float64)

    assert l.weight.dtype() == DType.Float64
    assert l.bias.dtype() == DType.Float64
    
    assert l.get_parameter('weight').dtype() == DType.Float64


def test_custom_mlp_integration():
    input_dim = 2
    hidden_dim = 4
    output_dim = 1
    batch_size = 3
    
    model = Mlp([input_dim, hidden_dim, output_dim])
    
    assert model.layer_count() == 2 # [2->4, 4->1]    
    x = Tensor.randn((batch_size, input_dim))    
    y = model.forward(x)
    
    expected_shape = [batch_size, output_dim]
    assert y.dims() == expected_shape

    gs = y.backward()
    first_layer_weight = model.linears[0].weight
    assert first_layer_weight in gs

