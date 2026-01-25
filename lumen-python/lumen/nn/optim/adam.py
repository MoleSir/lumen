from .. import Parameter
from ... import GradStore, Tensor
from .optimizer import Optimizer
from typing import Iterable, Dict, Tuple


class Adam(Optimizer):
    def __init__(
        self, 
        params: Iterable[Parameter], 
        learning_rate: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8
    ):
        
        super().__init__(params)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.moment: Dict[Parameter, Tuple[Tensor, Tensor]] = {} 
        self.t = 0  

    def step(self, grads: GradStore):
        self.t += 1
        for param in self.params:
            grad = grads[param]
            if grad is None:
                continue

            if param not in self.moment:
                self.moment[param] = (Tensor.zeros_like(param), Tensor.zeros_like(param))

            m, v = self.moment[param]

            # m = beta1 * m + (1 - beta1) * grad
            m.mul_(self.beta1).add_((1 - self.beta1) * grad)
            # v = beta2 * v + (1 - beta2) * grad^2
            v.mul_(self.beta2).add_((1 - self.beta2) * grad * grad)

            # Bias Correction)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # param = param - lr * m_hat / (sqrt(v_hat) + eps)
            adjusted_grad = m_hat / (v_hat.sqrt() + self.eps)
            param.sub_(self.learning_rate * adjusted_grad)