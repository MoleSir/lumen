from typing import Any, Callable
import functools
from ._lumen import (
    Tensor,
    GradStore,
    NoGradGuard,
    DType,
    is_grad_enabled,
    set_grad_enabled
)


class no_grad:
    """
    Context-manager that disabled gradient calculation.
    
    Can be used as a context manager:
        with no_grad():
            ...
            
    Or as a decorator:
        @no_grad()
        def foo():
            ...
    """
    def __init__(self):
        self._guard = NoGradGuard()
    
    def __enter__(self):
        self._guard.lock()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self._guard.unlock()

    def __call__(self, func):
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper