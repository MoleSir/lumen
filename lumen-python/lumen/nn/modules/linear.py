from dataclasses import dataclass
from typing import Optional
import math
from ... import Tensor, DType
from .. import Init
from .. import functional as F
from .module import Module 


@dataclass
class LinearConfig:
    in_features: int
    out_features: int
    bias: bool
    dtype: Optional[DType]


class Linear(Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool, 
            dtype: Optional[DType]=None,
            init: Optional[Init]=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if dtype is None:
            dtype = DType.Float32

        if init is None:
            gain = 1.0 / math.sqrt(3)
            init = Init.kaiming_uniform(gain, False)    

        self.weight = init.init_param((out_features, in_features), dtype, in_features, out_features)
        if bias:
            self.bias = Init.zeros().init_param((out_features,), dtype=dtype)
        else:
            self.bias = None
        
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    @classmethod
    def init(cls, config: LinearConfig, init: Optional[Init] = None):
        return Linear(config.in_features, config.out_features, config.bias, dtype=config.dtype, init=init)