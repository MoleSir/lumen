from ... import Tensor
from .module import Module 

class GeluErf(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.gelu_erf()


class Gelu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.gelu()


class LeakyRelu(Module):
    def __init__(self, negative_slope: float):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, xs: Tensor) -> Tensor:
        return xs.leaky_rele(self.negative_slope)
    
    def extra_repr(self):
        return f'negative_slope={self.negative_slope}'


class Recip(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.recip()


class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.relu()
    

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.sigmoid()
    

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.tanh()