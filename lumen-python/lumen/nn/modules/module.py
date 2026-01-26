from collections import OrderedDict, namedtuple
from typing import Union, Tuple, Any, Callable, TypeVar, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from .. import Parameter, Buffer
from ... import Tensor, no_grad, DType


M = TypeVar('M', bound='Module')


class Module:
    """Base class for all neural network modules"""

    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._buffers: Dict[str, Buffer] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()

        self._forward_hooks: List[Callable] = []

    ##################################################################
    #              Register Buffer / Paramter / Module
    ##################################################################

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

    ##################################################################
    #              Get Buffer / Paramter / Module
    ##################################################################

    def get_parameter(self, target: str) -> "Parameter":
        """
        Returns the parameter given by ``target`` if it exists,
        otherwise throws an error.
        """
        module_path, _, param_name = target.rpartition(".")

        mod = self.get_module(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(f'{mod._class_name()} has no attribute `{param_name}`')
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
            raise AttributeError(f'{mod._class_name()} has no attribute `{buffer_name}`')
        buffer = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError(f"`{buffer_name}` is not a buffer")

        return buffer

    def get_module(self, target: str) -> "Module":
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
                raise AttributeError(f'{mod._class_name()} has no attribute `{item}`')
            mod = getattr(mod, item)

        return mod
    
    ##################################################################
    #              Get Modules
    ##################################################################
    
    def modules(self) -> Iterator['Module']:
        """Returns an iterator over all modules in the network."""
        for _, module in self.named_modules():
            yield module

    def children_modules(self) -> Iterator['Module']:
        """Returns an iterator over immediate children modules."""
        for _, module in self.named_children_modules():
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

    def named_children_modules(self) -> Iterator[Tuple[str, 'Module']]:
        """
        Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        """
        for name, module in self._modules.items():
            yield name, module

    ##################################################################
    #              Get Params
    ##################################################################

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

    def parameter_count(self) -> int:
        return sum(1 for _ in self.parameters())
    
    def parameter_element_count(self) -> int:
        return sum(p.element_count() for p in self.parameters())

    ##################################################################
    #              Get Buffers
    ##################################################################

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

    def buffer_count(self) -> int:
        return sum(1 for _ in self.buffers())
    
    def buffer_element_count(self) -> int:
        return sum(p.element_count() for p in self.buffers())
    
    ##################################################################
    #              State dcit
    ##################################################################

    def state_dict(self) -> Iterator[Tuple[str, 'Tensor']]:
        for name, param in self.named_parameters():
            yield name, param
        for name, buffer in self.named_buffers():
            yield name, buffer        

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True):
        self._load_state_dict(state_dict, strict, '')
    
    def _load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool, prefix: str = ''):
        names = list(self._parameters.keys())
        for name in names:
            param_name = prefix + ('.' if prefix else '') + name
            src_tensor = state_dict.get(param_name)
            if src_tensor is None:
                if strict:
                    raise ValueError(f"Not Found tensor '{name}'")
            else:
                origin_param = self._parameters[name]
                self._parameters[name] = Parameter(src_tensor, requires_grad=origin_param.requires_grad())

        names = list(self._buffers.keys())
        for name in names:
            buffer_name = prefix + ('.' if prefix else '') + name
            src_tensor = state_dict.get(buffer_name)
            if src_tensor is None:
                if strict:
                    raise ValueError(f"Not Found tensor '{name}'")
            else:
                self._buffers[name] = Buffer(src_tensor)

        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            module._load_state_dict(state_dict, strict, submodule_prefix)

    ##################################################################
    #              Apply
    ##################################################################

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        self.training = mode
        for module in self.children_modules():
            module.train(mode)

    def eval(self):
        self.train(False)

    def float(self: M) -> M:
        return self.apply_tensor(lambda t: t.to_dtype(DType.Float32))
    
    def double(self: M) -> M:
        return self.apply_tensor(lambda t: t.to_dtype(DType.Float64))

    def to(self: M, dtype: DType) -> M:
        return self.apply_tensor(lambda t: t.to_dtype(dtype))

    def apply(self: M, fn: Callable[['Module'], None]) -> M:
        for module in self.children_modules():
            module.apply(fn)
        fn(self)
        return self
    
    def apply_tensor(self: M, fn: Callable[['Tensor'], 'Tensor']) -> M:
        for module in self.children_modules():
            module.apply_tensor(fn)
        
        for key, param in self._parameters.items():
            param_applied = fn(param)
            out_param = Parameter(param_applied, param.requires_grad())
            self._parameters[key] = out_param
        
        for key, buffer in self._buffers.items():
            buffer_applied = fn(buffer)
            out_buffer = Buffer(buffer_applied)
            self._buffers[key] = fn(out_buffer)

        return self
    
    ##################################################################
    #              Call
    ##################################################################

    def __call__(self, *args, **kwargs) -> Any:
        result = self.forward(*args, **kwargs)
        for hook in self._forward_hooks:
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result

        return result

    def register_forward_hook(self, hook: Callable):
        self._forward_hooks.append(hook)
        return hook 

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()
    
    def clear_forward_hook(self):
        self._forward_hooks.clear()

    ##################################################################
    #              Repr
    ##################################################################

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f'({key}): {mod_str}')
        lines = extra_lines + child_lines

        main_str = self._class_name() + '('
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        
        main_str += ')'
        return main_str

    def extra_repr(self) -> str:
        return ''
    
    ##################################################################
    #              Attr
    ##################################################################

    def __setattr__(self, name: str, value: Union[Parameter, Buffer, 'Module']) -> None:
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Buffer):
            self.register_buffer(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
      
        raise AttributeError(f"'{self._class_name()}' object has no attribute '{name}'")

    def _class_name(self):
        return self.__class__.__name__
    

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s