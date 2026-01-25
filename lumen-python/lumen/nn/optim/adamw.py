from .. import Parameter
from ... import GradStore, Tensor
from .optimizer import Optimizer
from typing import Iterable, Dict, Tuple
import math

class AdamW(Optimizer):
    def __init__(
        self, 
        params: Iterable[Parameter], 
        learning_rate: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.moment: Dict[Parameter, Tuple[Tensor, Tensor]] = {} 
        self.step_t = 0  

    def step(self, grads: GradStore):
        self.step_t += 1

        scale_m = 1 / (1 - math.pow(self.beta1, self.step_t))
        scale_v = 1 / (1 - math.pow(self.beta2, self.step_t))

        for param in self.params:
            grad = grads[param]
            if grad is None:
                continue

            if param not in self.moment:
                self.moment[param] = (Tensor.zeros_like(param), Tensor.zeros_like(param))

            m, v = self.moment[param]

            m.mul_(self.beta1).add_((1 - self.beta1) * grad)
            v.mul_(self.beta2).add_((1 - self.beta2) * grad * grad)

            m_hat = scale_m * m
            v_hat = scale_v * v

            adjusted_grad = m_hat / (v_hat.sqrt() + self.eps)

            param.sub_(self.learning_rate * self.weight_decay * param)
            param.sub_(self.learning_rate * adjusted_grad)