use lumen_core::{DynTensor, GradStore, NoGradGuard, TensorId};
use pyo3::{exceptions::PyValueError, prelude::*};

use super::PyTensor;

#[pyclass(name = "GradStore")]
#[derive(Clone)]
pub struct PyGradStore {
    pub inner: DynGradStore,
}

#[derive(Clone)]
pub enum DynGradStore {
    F32(GradStore<f32>),
    F64(GradStore<f64>),
}

#[pyclass]
pub struct PyNoGradGuard {
    guard: Option<NoGradGuard>,
}

impl Drop for PyNoGradGuard {
    fn drop(&mut self) {
        self.guard.take();
    }
}

#[pymethods]
impl PyNoGradGuard {
    #[new]
    fn new() -> Self {
        Self {
            guard: None,
        }
    }
    
    fn __enter__<'p>(mut slf: PyRefMut<'p, Self>) -> PyResult<PyRefMut<'p, Self>> {
        slf.guard = Some(NoGradGuard::new());
        Ok(slf)
    }
     
    fn __exit__(&mut self, _exc_type: &Bound<'_, PyAny>, _exc_value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.guard.take();
        Ok(false)
    }
}

#[pyfunction]
pub fn no_grad() -> PyNoGradGuard {
    PyNoGradGuard::new()
}

#[pyfunction]
pub fn set_grad_enabled(enabled: bool) {
    lumen_core::set_grad_enabled(enabled);
}

#[pyfunction]
pub fn is_grad_enabled() -> bool {
    lumen_core::is_grad_enabled()
}


#[pymethods]
impl PyGradStore {
    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Option<PyTensor>> {
        if let Ok(tensor) = index.extract::<PyTensor>() {
            match &self.inner {
                DynGradStore::F32(grads) => {
                    if let DynTensor::F32(tensor) = &tensor.inner {
                        Ok(grads.get(tensor).cloned().map(Into::into))
                    } else {
                        Err(PyValueError::new_err(format!("expect f32 tenspor, but got {:?}", tensor.dtype())))
                    }
                }
                DynGradStore::F64(grads) => {
                    if let DynTensor::F64(tensor) = &tensor.inner {
                        Ok(grads.get(tensor).cloned().map(Into::into))
                    } else {
                        Err(PyValueError::new_err(format!("expect f64 tenspor, but got {:?}", tensor.dtype())))
                    }
                }
            }
        } else if let Ok(id) = index.extract::<usize> () {
            match &self.inner {
                DynGradStore::F32(grads) => Ok(grads.get_by_index(id).cloned().map(Into::into)),
                DynGradStore::F64(grads) => Ok(grads.get_by_index(id).cloned().map(Into::into)),
            }
        } else {
            Err(PyValueError::new_err(format!("unsupport index: {}", index)))
        }

    }

    fn __contains__(&self, index: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.__getitem__(index).map(|opt| opt.is_some())
    }

    fn __iter__(&self) -> PyGradStoreIter {
        let items: Vec<(TensorId, DynTensor)> = match &self.inner {
            DynGradStore::F32(store) => {
                store.iter()
                    .map(|(k, v)| (*k, DynTensor::F32(v.clone())))
                    .collect()
            },
            DynGradStore::F64(store) => {
                store.iter()
                    .map(|(k, v)| (*k, DynTensor::F64(v.clone())))
                    .collect()
            },
        };

        PyGradStoreIter {
            iter: items.into_iter(),
        }
    }

    fn items(&self) -> PyGradStoreIter {
        self.__iter__()
    }

    fn keys(&self) -> Vec<usize> {
        match &self.inner {
            DynGradStore::F32(store) => store.get_ids().map(|k| k.value()).collect(),
            DynGradStore::F64(store) => store.get_ids().map(|k| k.value()).collect(),
        }
    }

    fn values(&self) -> Vec<PyTensor> {
        match &self.inner {
            DynGradStore::F32(store) => {
                store.tensors()
                    .map(|v| PyTensor { inner: DynTensor::F32(v.clone()) })
                    .collect()
            },
            DynGradStore::F64(store) => {
                store.tensors()
                    .map(|v| PyTensor { inner: DynTensor::F64(v.clone()) })
                    .collect()
            },
        }
    }
    
    fn __len__(&self) -> usize {
        match &self.inner {
            DynGradStore::F32(store) => store.len(),
            DynGradStore::F64(store) => store.len(),
        }
    }
}

#[pyclass]
pub struct PyGradStoreIter {
    iter: std::vec::IntoIter<(TensorId, DynTensor)>,
}

#[pymethods]
impl PyGradStoreIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<(usize, PyTensor)> {        
        slf.iter.next().map(|(id, dyn_tensor)| {
            let py_id = id.value();            
            let py_tensor = PyTensor { inner: dyn_tensor };
            
            (py_id, py_tensor)
        })
    }
}