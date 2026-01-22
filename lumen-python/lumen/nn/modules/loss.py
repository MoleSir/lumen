from typing import Optional
import math
from ... import Tensor, DType
from .. import Init
from .. import functional as F
from .module import Module 


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
    

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy_indices(input, target)
    

class NLLLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target)