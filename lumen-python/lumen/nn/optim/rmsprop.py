from .. import Parameter
from ... import GradStore, Tensor
from .optimizer import Optimizer
from typing import Iterable, Mapping


class RMSprop(Optimizer):
    def __init__(
            self, 
            params: Iterable[Parameter], 
            learning_rate: float = 1e-2, 
            alpha: float = 0.99, 
            eps: float = 1e-8
        ):
        super().__init__(params)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps
        self.square_avg: Mapping[Tensor, Tensor] = { param: Tensor.zeros_like(param) for param in self.params }

    def step(self, grads: GradStore):
        for param in self.params:
            grad = grads[param]
            if grad is None:
                continue

            s = self.square_avg[param]
            # s = alpha * s + (1 - alpha) * grad^2
            s.mul_(self.alpha).add_((1.0 - self.alpha) * grad * grad)

            # param = param - lr * grad / (sqrt(s) + eps)
            avg = s.sqrt() + self.eps
            param.sub_(self.learning_rate * grad / avg)