from ... import Tensor
from .. import functional as F
from .module import Module 


class Dropout(Module):
    def __init__(self, drop_p: float):
        super().__init__()
        self.drop_p = drop_p

    def forward(self, xs: Tensor) -> Tensor:
        if self.training:
            return F.dropout(xs, self.drop_p)
        else:
            return xs