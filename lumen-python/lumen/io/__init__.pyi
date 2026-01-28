from typing import Mapping, Optional, Tuple, Dict
from .. import Tensor


def load_safetensors_file(path: str) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
    ...


def save_safetensors_file(tensors: Mapping[str, Tensor], path: str, metadata: Optional[Mapping[str, str]]): 
    ...