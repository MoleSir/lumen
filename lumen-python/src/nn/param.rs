use pyo3::prelude::*;
use crate::core::PyTensor;

#[pyclass(name = "Parameter", extends=PyTensor)]
#[derive(Clone)]
pub struct PyParameter {}

#[pymethods]
impl PyParameter {
    #[new]
    #[pyo3(signature = (data, requires_grad=true))]
    pub fn new(data: &PyTensor, requires_grad: bool) -> PyResult<(Self, PyTensor)> {        
        let new_tensor_base = data.clone(); 
        new_tensor_base.detach()?.set_requires_grad(requires_grad)?;
        Ok((PyParameter {}, new_tensor_base))
    }
}

#[pyclass(name = "Buffer", extends=PyTensor)]
#[derive(Clone)]
pub struct PyBuffer {}

#[pymethods]
impl PyBuffer {
    #[new]
    pub fn new(data: &PyTensor) -> PyResult<(Self, PyTensor)> {        
        let new_tensor_base = data.clone(); 
        new_tensor_base.detach()?.set_requires_grad(false)?;
        Ok((PyBuffer {}, new_tensor_base))
    }
}