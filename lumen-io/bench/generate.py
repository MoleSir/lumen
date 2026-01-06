import torch
from safetensors.torch import save_file

tensors = {
    "embedding": torch.zeros((2, 2)),
    "attention": torch.rand((3, 3), dtype=torch.float32),
    "bias": torch.tensor([1, 2, 3], dtype=torch.int32) 
}

save_file(tensors, "test.safetensors")

