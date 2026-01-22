from .._lumen import nn as rust_nn 

Init = rust_nn.Init
Parameter = rust_nn.Parameter
Buffer = rust_nn.Buffer


def init_param(self, shape, dtype, fan_in=None, fan_out=None):
    t = self.init(shape, dtype, fan_in, fan_out)
    return Parameter(t)

def init_buffer(self, shape, dtype, fan_in=None, fan_out=None):
    t = self.init(shape, dtype, fan_in, fan_out)
    return Buffer(t)

Init.init_param = init_param
Init.init_buffer = init_buffer

del init_param, init_buffer