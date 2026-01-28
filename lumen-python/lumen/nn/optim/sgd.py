from typing import Iterable
from .. import Parameter
from ... import GradStore
from .optimizer import Optimizer


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum)."""
    def __init__(self, params: Iterable[Parameter], learning_rate: float):
        super().__init__(params)
        self.learning_rate = learning_rate

    def step(self, grads: GradStore):
        for param in self.params:
            grad = grads[param]
            if grad is not None:
                param.sub_(self.learning_rate * grad)