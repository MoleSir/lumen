use std::collections::HashMap;
use pyo3::prelude::*;
use crate::core::{to_runtime_error, PyTensor};

#[pyfunction]
pub fn load_safetensors_file(path: &str) -> PyResult<(HashMap<String, PyTensor>, Option<HashMap<String, String>>)> {
    let content = lumen_io::safetensors::load_file(path).map_err(to_runtime_error)?;
    let tensors = content.tensors.into_iter().map(|(name, t)| (name, PyTensor::from_inner(t))).collect();
    Ok((tensors, content.metadata))
}

#[pyfunction]
#[pyo3(signature = (tensors, path, metadata=None))]
pub fn save_safetensors_file(tensors: HashMap<String, PyTensor>, path: &str, metadata: Option<HashMap<String, String>>) -> PyResult<()> {
    let tensors = tensors.into_iter().map(|(name, t)| (name, t.inner)).collect();
    lumen_io::safetensors::save_file(&tensors, metadata.as_ref(), path).map_err(to_runtime_error)?;
    Ok(())
}
