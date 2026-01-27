from .dataset import Dataset
from .. import Tensor
from typing import List, TypeVar, Callable, Any, Generic, Tuple
import math
import random


Item = TypeVar('Item', covariant=True)
Batch = TypeVar('Batch', covariant=True)


def default_batch(items: List[Item]) -> Batch:
    assert len(items) != 0, "null batch"
    
    elem = items[0]

    if isinstance(elem, Tensor):
        return Tensor.stack(items, 0)
    
    elif isinstance(elem, (tuple, list)):
        # [(img1, lab1), (img2, lab2), (img3, lab3)] => ((img1, img2, img3), (lab1, lab2, lab3))
        transposed = zip(*items) 
        return tuple(default_batch(list(samples)) for samples in transposed)

    else:
        raise RuntimeError(f'default_batch not support {type(elem)}, you need define a batch_fn')


class DataLoader(Generic[Item, Batch]):
    def __init__(
            self, 
            dataset: Dataset[Item], 
            batch_size: int = 1,
            shuffle: bool = False,
            batch_fn: Callable[[List[Item]], Batch] = default_batch,
            drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_fn = batch_fn
        self.drop_last = drop_last

    def __len__(self) -> int:
        l = len(self.dataset)
        if self.drop_last:
            return l // self.batch_size
        else:
            return math.ceil(l / self.batch_size)

    def dataset_len(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> 'DataLoaderIter[Item, Batch]':
        return DataLoaderIter(self, self._get_iter_indices())

    def _get_iter_indices(self) -> List[int]:
        indices = list(range(self.dataset_len()))
        
        if self.shuffle:
            random.shuffle(indices)
            
        return indices


class DataLoaderIter(Generic[Item, Batch]):
    def __init__(self, loader: DataLoader[Item, Batch], indices: List[int]):
        self.loader = loader
        self.indices = indices
        self.cursor = 0

    def __iter__(self) -> 'DataLoaderIter[Item, Batch]':
        return self

    def __next__(self) -> Batch:
        begin = self.cursor
        total_len = self.loader.dataset_len()
        
        if begin >= total_len:
            raise StopIteration
        
        end = begin + self.loader.batch_size
        if self.loader.drop_last and end > total_len:
            raise StopIteration

        real_end = min(end, total_len)
        batch_indices = self.indices[begin:real_end]
        items = [self.loader.dataset[i] for i in batch_indices]
        
        self.cursor = end
        
        batch = self.loader.batch_fn(items)
        return batch
    