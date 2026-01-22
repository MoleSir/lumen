from collections import OrderedDict, namedtuple
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from .. import Parameter, Buffer


class Module:
    """Base class for all neural network modules"""

    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._buffers: Dict[str, Buffer] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()

    def register_buffer(self, name: str, buffer: Buffer) -> None:
        """Adds a buffer to the module."""
        if name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        if '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        if not isinstance(buffer, Buffer):
            raise TypeError(f"cannot assign '{type(buffer)}' object to buffer '{name}' (Buffer)")
        
        self._buffers[name] = buffer

    def register_parameter(self, name: str, param: Parameter) -> None:
        """Adds a parameter to the module."""
        if '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        if name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        if not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param)}' object to buffer '{name}' (Parameter)")

        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module') -> None:
        """Adds a child module to the current module."""
        if '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        if name == '':
            raise KeyError("module name can't be empty string \"\"")
        if not isinstance(module, Module):
            raise TypeError(f"{type(module)} is not a Module subclass")

        self._modules[name] = module

    def get_parameter(self, target: str) -> "Parameter":
        """
        Returns the parameter given by ``target`` if it exists,
        otherwise throws an error.
        """
        module_path, _, param_name = target.rpartition(".")

        mod = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(f'{mod._get_name()} has no attribute `{param_name}`')
        param = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError(f"`{param_name}` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Parameter":
        """
        Returns the buffer given by ``target`` if it exists,
        otherwise throws an error.
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(f'{mod._get_name()} has no attribute `{buffer_name}`')
        buffer = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError(f"`{buffer_name}` is not a buffer")

        return buffer

    def get_submodule(self, target: str) -> "Module":
        """
        Returns the submodule given by ``target`` if it exists, 
        otherwise throws an error.
        """
        if target == "":
            return self
        
        atoms: List[str] = target.split('.')
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(f'{mod._get_name()} has no attribute `{item}`')
            mod = getattr(mod, item)

        return mod
    
    def modules(self) -> Iterator['Module']:
        """Returns an iterator over all modules in the network."""
        for _, module in self.named_modules():
            yield module

    def children(self) -> Iterator['Module']:
        """Returns an iterator over immediate children modules."""
        for _, module in self.named_children():
            yield module

    def named_modules(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.
        """
        return self._named_modules('')

    def _named_modules(self, prefix: str) -> Iterator[Tuple[str, 'Module']]:
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for m in module._named_modules(submodule_prefix):
                yield m

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        """
        for name, module in self._modules.items():
            yield name, module

    def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
        """
        Returns an iterator over module parameters
        """
        return self._named_parameters('')

    def _named_parameters(self, prefix: str) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            param_name = prefix + ('.' if prefix else '') + name
            yield param_name, param

        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for ps in module._named_parameters(submodule_prefix):
                yield ps

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self) -> Iterator[Tuple[str, Buffer]]:
        """
        Returns an iterator over module buffer
        """
        return self._named_buffers('')

    def _named_buffers(self, prefix: str) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._buffers.items():
            buffer_name = prefix + ('.' if prefix else '') + name
            yield buffer_name, param

        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            for bs in module._named_buffers(submodule_prefix):
                yield bs

    def buffers(self) -> Iterator[Buffer]:
        for _, p in self.named_buffers():
            yield p

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)

    def eval(self):
        self.train(False)

    def __setattr__(self, name: str, value: Union[Parameter, 'Module']) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        if isinstance(value, Buffer):
            self.register_buffer(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)