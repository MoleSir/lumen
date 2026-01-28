from ... import Tensor
from .. import functional as F
from .module import Module 


class Loss(Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction


class MSELoss(Loss):
    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, self.reduction)
    

class L1Loss(Loss):
    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, self.reduction)
    

class CrossEntropyLoss(Loss):
    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy_indices(input, target)
    

class NllLoss(Loss):
    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target)