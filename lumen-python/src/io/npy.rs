use std::collections::HashMap;
use pyo3::prelude::*;
use crate::core::{to_runtime_error, PyTensor};

#[pyfunction]
pub fn load_npy_file(path: &str) -> PyResult<PyTensor> {
    let tensor = lumen_io::npy::load_npy_file(path).map_err(to_runtime_error)?;
    Ok(PyTensor::from_inner(tensor))
}

#[pyfunction]
pub fn load_npz_file(path: &str) -> PyResult<HashMap<String, PyTensor>> {
    let tensors = lumen_io::npy::load_npz_file(path).map_err(to_runtime_error)?;
    Ok(tensors.into_iter().map(|(n, t)| (n, PyTensor::from_inner(t))).collect())
} 

#[pyfunction]
pub fn save_npy_file(tensor: &PyTensor, path: &str) -> PyResult<()> {
    lumen_io::npy::save_npy_file(tensor.inner.clone(), path).map_err(to_runtime_error)?;
    Ok(())
}

#[pyfunction]
pub fn save_npz_file(tensors: HashMap<String, PyTensor>, path: &str) -> PyResult<()> {
    let tensors = tensors.into_iter().map(|(name, t)| (name, t.inner)).collect();
    lumen_io::npy::save_npz_file(&tensors, path).map_err(to_runtime_error)?;
    Ok(())
}