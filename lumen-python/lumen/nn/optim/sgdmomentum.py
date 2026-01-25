from .. import Parameter
from ... import GradStore, Tensor
from .optimizer import Optimizer
from typing import Iterable, Mapping


class SGDMomentum(Optimizer):
    def __init__(self, params: Iterable[Parameter], learning_rate: float, momentum: float = 0.9):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities: Mapping[Tensor, Tensor] = { param: Tensor.zeros_like(param) for param in self.params }

    def step(self, grads: GradStore):
        for param in self.params:
            grad = grads[param]
            if grad is None:
                continue

            v = self.velocities[param]
            v.mul_(self.momentum).add_(grad)
            
            param.sub_(self.learning_rate * v)