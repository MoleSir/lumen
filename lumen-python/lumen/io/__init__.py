from .._lumen import io as rust_io 

load_safetensors_file = rust_io.load_safetensors_file
save_safetensors_file = rust_io.save_safetensors_file
load_npy_file = rust_io.load_npy_file
load_npz_file = rust_io.load_npz_file
save_npy_file = rust_io.save_npy_file
save_npz_file = rust_io.save_npz_file