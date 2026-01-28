from typing import Literal, Tuple, List, TypeAlias, Optional
from pathlib import Path
import gzip
import shutil
import urllib.request
import os
from .. import Dataset, DataLoader
from ... import Tensor, DType


URL: str = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# URL: str = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES: str = "train-images-idx3-ubyte"
TRAIN_LABELS: str = "train-labels-idx1-ubyte"
TEST_IMAGES: str = "t10k-images-idx3-ubyte"
TEST_LABELS: str = "t10k-labels-idx1-ubyte"

WIDTH: int = 28
HEIGHT: int = 28
NUM_PIXELS: int = WIDTH * HEIGHT


MnistItem: TypeAlias = Tuple[List[float], int]
MnistBatch: TypeAlias = Tuple[Tensor, Tensor]


class MnistDataset(Dataset[MnistItem]):
    def __init__(self, split: Literal['train', 'test'], cache_dir: Optional[str] = None):
        super().__init__()
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "lumen-dataset"
        else:
            cache_dir = Path(cache_dir)

        images_path, label_path = MnistDataset.download(split, cache_dir)
        
        self.images = MnistDataset.read_images(images_path)
        self.labels = MnistDataset.read_labels(label_path)

    def __getitem__(self, index: int) -> Tuple[List[float], int]:
        if index >= len(self.labels):
            raise IndexError(f"Index {index} out of range")
        return (self.images[index], self.labels[index])

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def read_images(images_path: str) -> List[List[float]]:
        with open(images_path, 'rb') as file:
            # read size
            file.seek(4, 0)
            num_images = int.from_bytes(file.read(4), byteorder='big')

            # read data
            file.seek(16, 0)
            raw_data = file.read(num_images * NUM_PIXELS)

            images = []
            scale = 1.0 / 255.0
            for i in range(num_images):
                start = i * NUM_PIXELS
                end = start + NUM_PIXELS
                pixel_data = [b * scale for b in raw_data[start:end]]
                images.append(pixel_data)

            return images
        
    @staticmethod
    def read_labels(label_path: str) -> List[int]:
        with open(label_path, 'rb') as file:
            # read size
            file.seek(4, 0)
            size = int.from_bytes(file.read(4), byteorder='big')
            
            # read data
            file.seek(8, 0)
            data = file.read(size)
            labels = list(map(int, data))

            return labels

    @staticmethod
    def download(split: Literal['train', 'test'], cache_dir: Path) -> Tuple[str, str]:
        split_dir =  cache_dir / 'mnist' / split
        os.makedirs(split_dir, exist_ok=True)

        if split == 'train':
            img_name, lbl_name = TRAIN_IMAGES, TRAIN_LABELS
        elif split == 'test':
            img_name, lbl_name = TEST_IMAGES, TEST_LABELS
        else:
            raise ValueError(f"Unknown split: {split}")

        images_path = MnistDataset.download_file(img_name, split_dir)
        label_path = MnistDataset.download_file(lbl_name, split_dir)

        return images_path, label_path
            
    @staticmethod
    def download_file(name: str, dest_dir: Path) -> str:
        file_path = dest_dir / name

        if not os.path.exists(file_path):
            try:
                with urllib.request.urlopen(f'{URL}{name}.gz') as response:
                    with gzip.GzipFile(fileobj=response) as uncompressed_stream:
                        with open(file_path, 'wb') as out_file:
                            shutil.copyfileobj(uncompressed_stream, out_file)
                            
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
            pass
        return file_path
    

class MnistDataLoader(DataLoader[MnistItem, TypeAlias]):
    def __init__(
            self, 
            dataset: MnistDataset, 
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False):
        super().__init__(dataset, batch_size, shuffle, mnist_batch, drop_last)


def mnist_batch(items: List[MnistItem]) -> MnistBatch:
    batch_size = len(items)
        
    all_pixels = []
    all_labels = []
    
    for image, label in items:
        all_pixels.extend(image) 
        all_labels.append(label)
        
    images = Tensor(all_pixels).reshape((batch_size, WIDTH, HEIGHT))     
    labels = Tensor(all_labels, dtype=DType.UInt32).unsqueeze(1) 
    
    return images, labels
