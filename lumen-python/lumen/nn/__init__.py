import functools
from .._lumen import nn as rust_nn 

Init = rust_nn.Init
Parameter = rust_nn.Parameter
Buffer = rust_nn.Buffer
EmptyInitGuard = rust_nn.EmptyInitGuard

def init_param(self, shape, dtype, fan_in=None, fan_out=None):
    t = self.init(shape, dtype, fan_in, fan_out)
    return Parameter(t)

def init_buffer(self, shape, dtype, fan_in=None, fan_out=None):
    t = self.init(shape, dtype, fan_in, fan_out)
    return Buffer(t)

Init.init_param = init_param
Init.init_buffer = init_buffer

del init_param, init_buffer


class empty_init:
    def __init__(self):
        self._guard = EmptyInitGuard()

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