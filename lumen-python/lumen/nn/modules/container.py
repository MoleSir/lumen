from . import Module
from typing import Optional, Iterable


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
    