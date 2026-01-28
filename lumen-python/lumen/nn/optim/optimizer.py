from typing import Iterable
from .. import Parameter
from ... import GradStore


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Iterable[Parameter]):
        self.params = list(params)

    def step(self, grads: GradStore):
        raise NotImplementedError