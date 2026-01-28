from ... import Tensor
from .. import functional as F
from .module import Module 


class Softmax(Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, xs: Tensor) -> Tensor:
        F.softmax(xs, self.dim)