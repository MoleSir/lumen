from . import Module
from .. import Parameter
from typing import Optional, Iterable, Mapping, Tuple


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def extend(self, modules: Iterable[Module]):
        offset = len(self)
        for i, module in enumerate(modules):
            self.register_module(str(offset + i), module)
        
    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]
        
    def __len__(self) -> int:
        return len(self._modules)
    

class ModuleDict(Module):
    """Holds submodules in a dictionary."""
    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.register_module(key, module)

    def __len__(self) -> int:
        return len(self._modules)
    
    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def update(self, modules: Mapping[str, Module]) -> None:
        for n, m in enumerate(modules):
            self.register_module(n, m)
            
    def keys(self) -> Iterable[str]:
        return self._modules.keys()
    
    def items(self) -> Iterable[Tuple[str, Module]]:
        return self._modules.items()
    
    def values(self) -> Iterable[Module]:
        return self._modules.values()


class ParameterList(Module):
    """Holds parameters in a list."""

    def __init__(self, params: Optional[Iterable[Parameter]] = None) -> None:
        super().__init__()
        if params is not None:
            self.extend(params)

    def extend(self, params: Iterable[Parameter]):
        offset = len(self)
        for i, param in enumerate(params):
            self.register_parameter(str(offset + i), param)
        
    def __getitem__(self, idx: int) -> Parameter:
        return self._parameters[str(idx)]
        
    def __len__(self) -> int:
        return len(self._parameters)
    