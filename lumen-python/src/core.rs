use lumen_core::{DynTensor, GradStore, Indexer, NumDType, Tensor, TensorId};
use pyo3::{exceptions::{PyRuntimeError, PyTypeError, PyValueError}, prelude::*, types::{PyList, PySlice, PyTuple}};
use paste::paste;

#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    pub inner: DynTensor,
}

#[pyclass(name = "DType", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PyDType {
    Bool,
    Float32,
    Float64,
    Int32,
    UInt32,
    UInt8,
}

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

macro_rules! impl_contruct {
    ($method:ident, $dtype:ident, $shape:ident) => {
        let target_dtype = $dtype.unwrap_or(PyDType::Float32);
        return match target_dtype {
            PyDType::Bool => Tensor::<bool>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::Float32 => Tensor::<f32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::Float64 => Tensor::<f64>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::UInt32 => Tensor::<u32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::Int32 => Tensor::<i32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::UInt8 => Tensor::<u8>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
        }
    };
}

macro_rules! impl_varient_method {
    ($self:ident, $t:ident, $expr:expr) => {
        match &$self.inner {
            DynTensor::Bool($t) => $expr,
            DynTensor::F32($t) => $expr,
            DynTensor::F64($t) => $expr,
            DynTensor::U32($t) => $expr,
            DynTensor::I32($t) => $expr,
            DynTensor::U8($t) => $expr,
        }
    };
}

// lhs : Tensor or Scalar
macro_rules! impl_arith_binary_lhs {
    ($lhs:ident, $rhs:ident, $method:ident) => {
        paste! {
            if let Ok(lhs) = $lhs.extract::<PyTensor>() {
                match (&lhs.inner, &$rhs.inner) {
                    (DynTensor::F32(lhs), DynTensor::F32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::F64(lhs), DynTensor::F64(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U32(lhs), DynTensor::U32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::I32(lhs), DynTensor::I32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U8(lhs), DynTensor::U8(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    _ => Err(PyTypeError::new_err(format!("unsupport {} with {:?} and {:?}", stringify!($method), lhs.dtype(), $rhs.dtype())))
                }
            } else {
                match &$rhs.inner {
                    DynTensor::U8(rhs) => {
                        let scalar = $lhs.extract::<u8>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for f32 tensor"))?;                
                        Tensor:: [< scalar_ $method >](scalar, rhs).map_err(to_value_error).map(Into::into)
                    },
                    DynTensor::F32(rhs) => {
                        let scalar = $lhs.extract::<f32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float for f32 tensor"))?;                
                        Tensor:: [< scalar_ $method >](scalar, rhs).map_err(to_value_error).map(Into::into)
                    },
                    DynTensor::F64(rhs) => {
                        let scalar = $lhs.extract::<f64>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float for f64 tensor"))?;
                        Tensor:: [< scalar_ $method >](scalar, rhs).map_err(to_value_error).map(Into::into)
                    },
                    DynTensor::I32(rhs) => {
                        let scalar = $lhs.extract::<i32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for i32 tensor"))?;
                        Tensor:: [< scalar_ $method >](scalar, rhs).map_err(to_value_error).map(Into::into)
                    },
                    DynTensor::U32(rhs) => {
                        let scalar = $lhs.extract::<u32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for i32 tensor"))?;
                        Tensor:: [< scalar_ $method >](scalar, rhs).map_err(to_value_error).map(Into::into)
                    },
                    _ => Err(pyo3::exceptions::PyTypeError::new_err("Unsupported dtype for scalar add")),
                }
            }
        }
    };
}

macro_rules! impl_arith_binary {
    ($lhs:ident, $rhs:ident, $method:ident) => {
        if let Ok(rhs) = $rhs.extract::<PyTensor>() {
            match (&$lhs.inner, &rhs.inner) {
                (DynTensor::F32(lhs), DynTensor::F32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                (DynTensor::F64(lhs), DynTensor::F64(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                (DynTensor::U32(lhs), DynTensor::U32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                (DynTensor::I32(lhs), DynTensor::I32(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                (DynTensor::U8(lhs), DynTensor::U8(rhs)) => lhs.$method(rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                _ => Err(PyTypeError::new_err(format!("unsupport {} with {:?} and {:?}", stringify!($method), $lhs.dtype(), rhs.dtype())))
            }
        } else {
            match &$lhs.inner {
                DynTensor::U8(lhs) => {
                    let scalar = $rhs.extract::<u8>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for f32 tensor"))?;                
                    lhs.$method(scalar).map_err(to_value_error).map(Into::into)
                },
                DynTensor::F32(lhs) => {
                    let scalar = $rhs.extract::<f32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float for f32 tensor"))?;                
                    lhs.$method(scalar).map_err(to_value_error).map(Into::into)
                },
                DynTensor::F64(lhs) => {
                    let scalar = $rhs.extract::<f64>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or float for f64 tensor"))?;
                    lhs.$method(scalar).map_err(to_value_error).map(Into::into)
                },
                DynTensor::I32(lhs) => {
                    let scalar = $rhs.extract::<i32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for i32 tensor"))?;
                    lhs.$method(scalar).map_err(to_value_error).map(Into::into)
                },
                DynTensor::U32(lhs) => {
                    let scalar = $rhs.extract::<u32>().map_err(|_| pyo3::exceptions::PyTypeError::new_err("Expected Tensor or int for i32 tensor"))?;
                    lhs.$method(scalar).map_err(to_value_error).map(Into::into)
                },
                _ => Err(pyo3::exceptions::PyTypeError::new_err("Unsupported dtype for scalar add")),
            }
        }
    };
}

macro_rules! impl_arith_unary {
    ($t:ident, $method:ident) => {
        match &$t.inner {
            DynTensor::F32(t) => Ok(t.$method().into()),
            DynTensor::F64(t) => Ok(t.$method().into()),
            _ => Err(PyValueError::new_err(format!("dtype {} not support {:?}", stringify!($method), $t.dtype()))),
        }
    };
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn zeros(shape: Vec<usize>, dtype: Option<PyDType>) -> PyResult<Self> {
        impl_contruct!(zeros, dtype, shape);
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn ones(shape: Vec<usize>, dtype: Option<PyDType>) -> PyResult<Self> {
        impl_contruct!(ones, dtype, shape);
    }

    #[staticmethod]
    #[pyo3(signature = (shape, min=None, max=None, dtype=None))]
    fn rand(shape: Vec<usize>, min: Option<f32>, max: Option<f32>, dtype: Option<PyDType>) -> PyResult<Self> {
        let min = min.unwrap_or(0.0);
        let max = max.unwrap_or(1.0);
        let dtype = dtype.unwrap_or(PyDType::Float32);

        match dtype {
            PyDType::Float32 => Tensor::rand(min, max, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::Float64 => Tensor::rand(min.to_f64(), max.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            dtype => Err(PyValueError::new_err(format!("dtype {:?} no support rand", dtype)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, mean=None, std=None, dtype=None))]
    fn randn(shape: Vec<usize>, mean: Option<f32>, std: Option<f32>, dtype: Option<PyDType>) -> PyResult<Self> {
        let mean = mean.unwrap_or(0.0);
        let std = std.unwrap_or(1.0);
        let dtype = dtype.unwrap_or(PyDType::Float32);

        match dtype {
            PyDType::Float32 => Tensor::randn(mean, std, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            PyDType::Float64 => Tensor::rand(mean.to_f64(), std.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            dtype => Err(PyValueError::new_err(format!("dtype {:?} no support randn", dtype)))
        }
    }

    fn dims(&self) -> Vec<usize> {
        impl_varient_method!(self, t, t.dims().to_vec())
    }

    fn dtype(&self) -> PyDType {
        match &self.inner {
            DynTensor::Bool(_) => PyDType::Bool,
            DynTensor::F32(_) => PyDType::Float32,
            DynTensor::F64(_) => PyDType::Float64,
            DynTensor::U32(_) => PyDType::UInt32,
            DynTensor::I32(_) => PyDType::Int32,
            DynTensor::U8(_) => PyDType::UInt8,
        }
    }

    fn set_requires_grad(&self, mode: bool) -> PyResult<()> {
        match &self.inner {
            DynTensor::Bool(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::F32(t) => Ok(t.set_requires_grad(mode)),
            DynTensor::F64(t) => Ok(t.set_requires_grad(mode)),
            DynTensor::U32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::I32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::U8(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
        }
    }

    fn requires_grad(&self) -> PyResult<bool> {
        match &self.inner {
            DynTensor::Bool(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::F32(t) => Ok(t.requires_grad()),
            DynTensor::F64(t) => Ok(t.requires_grad()),
            DynTensor::U32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::I32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::U8(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
        }
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.add(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.add(lhs)
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.mul(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.mul(lhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.sub(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary_lhs!(lhs, self, sub)
    }

    fn __div__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.div(rhs)
    }

    fn __rdiv__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary_lhs!(lhs, self, div)
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, add)
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, sub)
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, mul)
    }

    fn div(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, div)
    }

    fn floor(&self) -> PyResult<Self> {
        impl_arith_unary!(self, floor)
    }

    fn ceil(&self) -> PyResult<Self> {
        impl_arith_unary!(self, ceil)
    }

    fn exp(&self) -> PyResult<Self> {
        impl_arith_unary!(self, exp)
    }

    fn ln(&self) -> PyResult<Self> {
        impl_arith_unary!(self, ln)
    }

    fn round(&self) -> PyResult<Self> {
        impl_arith_unary!(self, round)
    }

    fn sin(&self) -> PyResult<Self> {
        impl_arith_unary!(self, sin)
    }

    fn cos(&self) -> PyResult<Self> {
        impl_arith_unary!(self, cos)
    }

    fn tanh(&self) -> PyResult<Self> {
        impl_arith_unary!(self, tanh)
    }

    fn sqrt(&self) -> PyResult<Self> {
        impl_arith_unary!(self, sqrt)
    }

    fn sqr(&self) -> PyResult<Self> {
        impl_arith_unary!(self, sqr)
    }

    fn abs(&self) -> PyResult<Self> {
        impl_arith_unary!(self, abs)
    }

    fn recip(&self) -> PyResult<Self> {
        impl_arith_unary!(self, recip)
    }

    fn gelu(&self) -> PyResult<Self> {
        impl_arith_unary!(self, gelu)
    }

    fn gelu_erf(&self) -> PyResult<Self> {
        impl_arith_unary!(self, gelu_erf)
    }

    fn erf(&self) -> PyResult<Self> {
        impl_arith_unary!(self, erf)
    }

    fn relu(&self) -> PyResult<Self> {
        impl_arith_unary!(self, relu)
    }

    fn silu(&self) -> PyResult<Self> {
        impl_arith_unary!(self, silu)
    }

    fn sigmoid(&self) -> PyResult<Self> {
        impl_arith_unary!(self, sigmoid)
    }

    #[pyo3(signature = (other, rtol=None, atol=None))]
    fn allclose(&self, other: &Self, rtol: Option<f64>, atol: Option<f64>) -> PyResult<bool> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);
        
        match (&self.inner, &other.inner) {
            (DynTensor::F32(this), DynTensor::F32(other)) => Ok(this.allclose(other, rtol, atol)),
            (DynTensor::F64(this), DynTensor::F64(other)) => Ok(this.allclose(other, rtol, atol)),
            (DynTensor::U32(this), DynTensor::U32(other)) => Ok(this.allclose(other, rtol, atol)),
            (DynTensor::I32(this), DynTensor::I32(other)) => Ok(this.allclose(other, rtol, atol)),
            (DynTensor::U8(this), DynTensor::U8(other)) => Ok(this.allclose(other, rtol, atol)),
            (DynTensor::Bool(this), DynTensor::Bool(other)) => Ok(this.eq(other)),
            _ => Ok(false)
        }
    }

    fn backward(&self) -> PyResult<PyGradStore> {
        match &self.inner {
            DynTensor::Bool(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::F32(t) => {
                let grads = t.backward().map_err(to_value_error)?;
                Ok(PyGradStore { inner: DynGradStore::F32(grads) })
            },
            DynTensor::F64(t) => {
                let grads = t.backward().map_err(to_value_error)?;
                Ok(PyGradStore { inner: DynGradStore::F64(grads) })
            },
            DynTensor::U32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::I32(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
            DynTensor::U8(_) => Err(PyRuntimeError::new_err("bool tensor no grad!")),
        }
    }

    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut indexers = Vec::new();

        if let Ok(tuple) = index.cast::<PyTuple>() {
            for item in tuple {
                indexers.push(py_to_indexer(&item)?);
            }
        } else if let Ok(list) = index.cast::<PyList>() {
            for item in list {
                indexers.push(py_to_indexer(&item)?);
            }
        } else {
            indexers.push(py_to_indexer(index)?);
        }

        let res_inner = impl_varient_method!(self, t, t.indexes(&indexers).map_err(to_value_error).map(Into::into)?);

        Ok(PyTensor { inner: res_inner })
    }

    fn __str__(&self) -> String {
        impl_varient_method!(self, t, format!("{}", t))
    }

    fn __repr__(&self) -> String {
        impl_varient_method!(self, t, format!("Tensor({}, shape={:?}, dtype={:?})", t, t.shape(), t.dtype()))
    }
}

#[pymethods]
impl PyGradStore {
    fn __getitem__(&self, tensor: PyTensor) -> PyResult<Option<PyTensor>> {
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

fn py_to_indexer(obj: &Bound<'_, PyAny>) -> PyResult<Indexer> {
    if let Ok(idx) = obj.extract::<usize>() {
        return Ok(Indexer::Select(idx));
    }

    if let Ok(slice_obj) = obj.cast::<PySlice>() {
        let indices = slice_obj.indices(isize::MAX)?;
        if indices.start < 0 {
            return Err(PyValueError::new_err(format!("start {} < 0", indices.start)));
        }
        let start: usize = indices.start as usize;

        if indices.step < 0 {
            return Err(PyValueError::new_err(format!("step {} < 0", indices.step)));
        }
        let step: usize = indices.step as usize;

        return Ok(Indexer::Slice(lumen_core::Slice {
            start,
            end: Some(indices.stop),
            step,
        }));
    }
    
    Err(PyTypeError::new_err(format!("Unsupported index type: {}", obj.get_type())))
}

macro_rules! impl_convert_with_type {
    ($variant:ident, $inner:ty) => {
        impl From<Tensor<$inner>> for PyTensor {
            fn from(t: Tensor<$inner>) -> Self {
                PyTensor { inner: DynTensor::$variant(t) }
            }
        }
    };
}

impl_convert_with_type!(Bool, bool);
impl_convert_with_type!(F32, f32);
impl_convert_with_type!(F64, f64);
impl_convert_with_type!(I32, i32);
impl_convert_with_type!(U32, u32);
impl_convert_with_type!(U8, u8);


fn to_value_error(e: lumen_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[allow(unused)]
fn to_runtime_error(e: lumen_core::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}