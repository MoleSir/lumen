

from ... import Tensor
from .. import functional as F
from .module import Module 


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: Tensor) -> Tensor:
        return xs.sigmoid()