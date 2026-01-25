from .. import Tensor
import bisect
from typing import List, Generic, TypeVar, Iterator, Tuple, Iterable, Sequence
import random
import math


Item = TypeVar('Item', covariant=True)


class Dataset(Generic[Item]):
    """An abstract class representing a :class:`Dataset`""" 

    def __getitem__(self, index: int) -> Item:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def __iter__(self) -> 'DatasetIter[Item]':
        return DatasetIter(self)


class DatasetIter(Generic[Item]):
    def __init__(self, dataset: Dataset[Item]):
        self.dataset = dataset
        self.index = 0

    def __iter__(self) -> 'DatasetIter[Item]':
        return self
    
    def __next__(self) -> Item:
        if self.index >= len(self.dataset):
            raise StopIteration
        index = self.index
        self.index += 1
        return self.dataset[index]
        
        
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    """Dataset wrapping tensors."""
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].dim(0) == tensor.dim(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)
    
    def __len__(self):
        return self.tensors[0].dim(0)
    

class ConcatDataset(Dataset[Item]):
    """Dataset as a concatenation of multiple datasets."""
    
    def __init__(self, datasets: Iterable[Dataset[Item]]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        
        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_sizes[dataset_index - 1]
        
        return self.datasets[dataset_index][sample_index]
    
    @staticmethod
    def cumsum(sequence) -> List[int]:
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
    

class SubsetDataset(Dataset[Item]):
    def __init__(self, dataset: Dataset[Item], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)
    
    @staticmethod
    def new(dataset: Dataset[Item], indices: Sequence[int]) -> 'SubsetDataset[Item]':
        if any(map(lambda index: index >= len(dataset), indices)):
            raise ValueError("index of out range of dataset")
        return SubsetDataset(dataset, indices)
    
    @staticmethod
    def select_all(dataset: Dataset[Item]) -> 'SubsetDataset[Item]':
        SubsetDataset(dataset, list[range(len(dataset))])

    def slice(self, start: int, end: int) -> 'SubsetDataset[Item]':
        SubsetDataset.new(self, self.indices[start:end])


def random_split(dataset: Dataset[Item], ratio: float) -> Tuple[SubsetDataset[Item]]:
    if ratio < 0. or ratio > 1.:
        raise ValueError(f'invalid ratio {ratio}')
    
    length = len(dataset)
    indices = list(range(length))

    random.shuffle(indices)

    split_index = int(math.floor(length * ratio))
    indices1, indices2 = indices[:split_index], indices[split_index:]

    return SubsetDataset(dataset, indices1), SubsetDataset(dataset, indices2)