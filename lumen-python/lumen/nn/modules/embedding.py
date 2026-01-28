from dataclasses import dataclass
from typing import Optional
import math
from ... import Tensor, DType
from .. import Init
from .. import functional as F
from .module import Module 


@dataclass
class EmbeddingConfig:
    num_embedding: int
    embedding_size: int
    dtype: Optional[DType]


class Embedding(Module):
    def __init__(
            self, 
            num_embedding: int, 
            embedding_size: int, 
            dtype: Optional[DType]=None,
            init: Optional[Init]=None
    ):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_size = embedding_size

        if dtype is None:
            dtype = DType.Float32

        if init is None:
            init = Init.normal()

        self.weight = init.init_param((num_embedding, embedding_size), dtype)
        
    def forward(self, x: Tensor) -> Tensor:
        return F.embeddding(self.weight, x)
    
    def extra_repr(self):
        return f'num_embedding={self.num_embedding}, embedding_size={self.embedding_size}'
    
    @classmethod
    def init(cls, config: EmbeddingConfig, init: Optional[Init] = None):
        return Embedding(config.embedding_size, config.embedding_size, dtype=config.dtype, init=init)