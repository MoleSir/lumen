from typing import Mapping, Optional, Tuple, Dict
from .. import Tensor


def load_safetensors_file(path: str) -> Tuple[Dict[str, Tensor], Dict[str, str]]:
    ...


def save_safetensors_file(tensors: Mapping[str, Tensor], path: str, metadata: Optional[Mapping[str, str]]): 
    ...


def load_npy_file(path: str) -> Tensor:
    ...


def save_npy_file(tensor: Tensor, path: str): 
    ...
    

def load_npz_file(path: str) -> Dict[str, Tensor]:
    ...


def save_npz_file(tensors: Mapping[str, Tensor], path: str): 
    ...