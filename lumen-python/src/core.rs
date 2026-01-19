use lumen_core::{DType, DTypeConvert, DynTensor, GradStore, Indexer, NoGradGuard, NumDType, Shape, Slice, Tensor, TensorId, Var, WithDType, D};
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

impl PyDType {
    fn to_dtype(&self) -> DType {
        match self {
            Self::Bool => DType::Bool,
            Self::Float32 => DType::F32,
            Self::Float64 => DType::F64,
            Self::UInt32 => DType::U32,
            Self::Int32 => DType::I32,
            Self::UInt8 => DType::U8,
        }
    }
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

macro_rules! impl_contruct {
    ($method:ident, $dtype:ident, $shape:ident, $requires_grad:ident) => {
        let target_dtype = $dtype.unwrap_or(PyDType::Float32);
        if $requires_grad {
            return match target_dtype {
                PyDType::Float32 => Var::<f32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::Float64 => Var::<f64>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                _ => Err(PyRuntimeError::new_err(format!("{:?} tensor no grad!", target_dtype))),
            }
        } else {
            return match target_dtype {
                PyDType::Bool => Tensor::<bool>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::Float32 => Tensor::<f32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::Float64 => Tensor::<f64>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::UInt32 => Tensor::<u32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::Int32 => Tensor::<i32>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
                PyDType::UInt8 => Tensor::<u8>::$method($shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            }
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

macro_rules! impl_numdtype_varient_method {
    ($self:ident, $t:ident, $expr:expr, $msg:expr) => {
        match &$self.inner {
            DynTensor::Bool(_) => Err(PyValueError::new_err(format!("bool unsupport {}", $msg))),
            DynTensor::F32($t) => $expr,
            DynTensor::F64($t) => $expr,
            DynTensor::U32($t) => $expr,
            DynTensor::I32($t) => $expr,
            DynTensor::U8($t) => $expr,
        }
    };
}

macro_rules! impl_reduce {
    ($op:ident, $self:ident, $dim:ident, $keep_dim:ident) => {
        paste! {
            match $dim {
                Some(dim) => {
                    let dim = py_to_dim(dim);
                    if $keep_dim {
                        impl_numdtype_varient_method!($self, t, t.[<$op _keepdim>](dim).map_err(to_value_error).map(Into::into), stringify!($op))
                    } else {
                        impl_numdtype_varient_method!($self, t, t.$op(dim).map_err(to_value_error).map(Into::into), stringify!($op))
                    }
                }
                None => {
                    impl_numdtype_varient_method!($self, t, t.[<$op _all>]().map_err(to_value_error).map(Into::into), stringify!($op))
                }
            }
        }
    };
}

// lhs : Tensor or Scalar
macro_rules! impl_arith_binary_lhs {
    ($lhs:ident, $rhs:ident, $method:ident) => {
        paste! {
            if let Ok(lhs) = $lhs.extract::<PyTensor>() {
                match (&lhs.inner, &$rhs.inner) {
                    (DynTensor::F32(lhs), DynTensor::F32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::F64(lhs), DynTensor::F64(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U32(lhs), DynTensor::U32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::I32(lhs), DynTensor::I32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U8(lhs), DynTensor::U8(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
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
        paste! {
            if let Ok(rhs) = $rhs.extract::<PyTensor>() {
                match (&$lhs.inner, &rhs.inner) {
                    (DynTensor::F32(lhs), DynTensor::F32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::F64(lhs), DynTensor::F64(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U32(lhs), DynTensor::U32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::I32(lhs), DynTensor::I32(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U8(lhs), DynTensor::U8(rhs)) => lhs.[<broadcast_ $method>](rhs).map_err(to_value_error).map(Into::<PyTensor>::into),
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
    #[new]
    #[pyo3(signature = (to_tensor, dtype=None, requires_grad=false))]
    fn __init__(to_tensor: &Bound<'_, PyAny>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        Self::new(to_tensor, dtype, requires_grad)
    }

    #[staticmethod]
    #[pyo3(signature = (to_tensor, dtype=None, requires_grad=false))]
    fn new(to_tensor: &Bound<'_, PyAny>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        let mut tensor = Self::new_impl(to_tensor)?;
        if let Some(dtype) = dtype {
            tensor = tensor.to_dtype(dtype);
        }
        if requires_grad {
            tensor.set_requires_grad(true)?;
        }

        Ok(tensor)
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, requires_grad=false))]
    fn zeros(shape: &Bound<'_, PyAny>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        impl_contruct!(zeros, dtype, shape, requires_grad);
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, requires_grad=false))]
    fn ones(shape: &Bound<'_, PyAny>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        impl_contruct!(ones, dtype, shape, requires_grad);
    }

    #[staticmethod]
    #[pyo3(signature = (shape, min=None, max=None, dtype=None, requires_grad=false))]
    fn rand(shape: &Bound<'_, PyAny>, min: Option<f32>, max: Option<f32>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        let min = min.unwrap_or(0.0);
        let max = max.unwrap_or(1.0);
        let dtype = dtype.unwrap_or(PyDType::Float32);

        match (dtype, requires_grad) {
            (PyDType::Float32, true) => Var::rand(min, max, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float32, false) => Tensor::rand(min, max, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float64, true) => Var::rand(min.to_f64(), max.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float64, false) => Tensor::rand(min.to_f64(), max.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (dtype, _) => Err(PyValueError::new_err(format!("dtype {:?} no support rand", dtype)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, mean=None, std=None, dtype=None, requires_grad=false))]
    fn randn(shape: &Bound<'_, PyAny>, mean: Option<f32>, std: Option<f32>, dtype: Option<PyDType>, requires_grad: bool) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        
        let mean = mean.unwrap_or(0.0);
        let std = std.unwrap_or(1.0);
        let dtype = dtype.unwrap_or(PyDType::Float32);

        match (dtype, requires_grad) {
            (PyDType::Float32, true) => Var::randn(mean, std, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float32, false) => Tensor::randn(mean, std, shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float64, true) => Var::rand(mean.to_f64(), std.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (PyDType::Float64, false) => Tensor::rand(mean.to_f64(), std.to_f64(), shape).map_err(to_value_error).map(Into::<PyTensor>::into),
            (dtype, _) => Err(PyValueError::new_err(format!("dtype {:?} no support randn", dtype)))
        }
    }

    #[staticmethod]
    fn trues(shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        Tensor::trues(shape).map_err(to_value_error).map(Into::<PyTensor>::into)
    }

    #[staticmethod]
    fn falses(shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        Tensor::falses(shape).map_err(to_value_error).map(Into::<PyTensor>::into)
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

    fn to_dtype(&self, dtype: PyDType) -> Self {
        impl_varient_method!(self, t, to_dtype(t, dtype))
    }

    fn item(&self) -> PyResult<Self> {
        impl_varient_method!(self, t, t.item().map_err(to_value_error).map(Into::into))
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

    fn eq(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, eq)
    }

    fn ne(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, ne)
    }

    fn le(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, le)
    }

    fn ge(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, ge)
    }

    fn lt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, lt)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        impl_arith_binary!(self, rhs, gt)
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

    #[pyo3(signature = (dim=None, keep_dim=false))]
    fn sum(&self, dim: Option<isize>, keep_dim: bool) -> PyResult<Self> {
        return impl_reduce!(sum, self, dim, keep_dim);
    }

    #[pyo3(signature = (dim=None, keep_dim=false))]
    fn min(&self, dim: Option<isize>, keep_dim: bool) -> PyResult<Self> {
        return impl_reduce!(min, self, dim, keep_dim);
    }

    #[pyo3(signature = (dim=None, keep_dim=false))]
    fn max(&self, dim: Option<isize>, keep_dim: bool) -> PyResult<Self> {
        return impl_reduce!(max, self, dim, keep_dim);
    }

    #[pyo3(signature = (dim=None, keep_dim=false))]
    fn mean(&self, dim: Option<isize>, keep_dim: bool) -> PyResult<Self> {
        return impl_reduce!(mean, self, dim, keep_dim);
    }

    #[pyo3(signature = (dim=None, keep_dim=false, unbiased=true))]
    fn var(&self, dim: Option<isize>, keep_dim: bool, unbiased: bool) -> PyResult<Self> {
        if unbiased {
            match dim {
                Some(dim) => {
                    let dim = py_to_dim(dim);
                    if keep_dim {
                        impl_numdtype_varient_method!(self, t, t.var_unbiased_keepdim(dim).map_err(to_value_error).map(Into::into), "var")
                    } else {
                        impl_numdtype_varient_method!(self, t, t.var_unbiased(dim).map_err(to_value_error).map(Into::into), stringify!($op))
                    }
                }
                None => {
                    impl_numdtype_varient_method!(self, t, t.var_unbiased_all().map_err(to_value_error).map(Into::into), stringify!($op))
                }
            }
        } else {
            match dim {
                Some(dim) => {
                    let dim = py_to_dim(dim);
                    if keep_dim {
                        impl_numdtype_varient_method!(self, t, t.var_keepdim(dim).map_err(to_value_error).map(Into::into), "var")
                    } else {
                        impl_numdtype_varient_method!(self, t, t.var(dim).map_err(to_value_error).map(Into::into), stringify!($op))
                    }
                }
                None => {
                    impl_numdtype_varient_method!(self, t, t.var_all().map_err(to_value_error).map(Into::into), stringify!($op))
                }
            }
        }
    }

    fn neg(&self) -> PyResult<Self> {
        match &self.inner {
            DynTensor::F32(t) => Ok(t.neg().into()),
            DynTensor::F64(t) => Ok(t.neg().into()),
            DynTensor::I32(t) => Ok(t.neg().into()),
            DynTensor::U32(_) => Err(PyValueError::new_err("u32 tensor not support neg")),
            DynTensor::U8(_) => Err(PyValueError::new_err("u8 tensor not support neg")),
            DynTensor::Bool(_) => Err(PyValueError::new_err("bool tensor not support neg")),
        }
    }

    fn __matmul__(&self, rhs: &PyTensor) -> PyResult<Self> {
        self.matmul(rhs)
    }

    fn __rmatmul__(&self, lhs: &PyTensor) -> PyResult<Self> {
        lhs.matmul(self)
    }

    fn matmul(&self, rhs: &PyTensor) -> PyResult<Self> {
        match (&self.inner, &rhs.inner) {
            (DynTensor::F32(lhs), DynTensor::F32(rhs)) => lhs.matmul(rhs).map_err(to_value_error).map(Into::into),
            (DynTensor::F64(lhs), DynTensor::F64(rhs)) => lhs.matmul(rhs).map_err(to_value_error).map(Into::into),
            (DynTensor::U32(lhs), DynTensor::U32(rhs)) => lhs.matmul(rhs).map_err(to_value_error).map(Into::into),
            (DynTensor::I32(lhs), DynTensor::I32(rhs)) => lhs.matmul(rhs).map_err(to_value_error).map(Into::into),
            (DynTensor::U8(lhs), DynTensor::U8(rhs)) =>  lhs.matmul(rhs).map_err(to_value_error).map(Into::into),
            _ => Err(PyTypeError::new_err(format!("unsupport matmul with {:?} and {:?}", self.dtype(), rhs.dtype())))
        } 
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

    fn squeeze(&self, dim: isize) -> PyResult<Self> {
        let dim = py_to_dim(dim);
        return impl_varient_method!(self, t, t.squeeze(dim).map_err(to_value_error).map(Into::into));
    }

    fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let dim = py_to_dim(dim);    
        return impl_varient_method!(self, t, t.unsqueeze(dim).map_err(to_value_error).map(Into::into));
    }

    fn narrow(&self, dim: isize, start: usize, len: usize) -> PyResult<Self> {
        let dim = py_to_dim(dim);    
        return impl_varient_method!(self, t, t.narrow(dim, start, len).map_err(to_value_error).map(Into::into));
    }

    fn slice(&self, dim: isize, slice: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dim = py_to_dim(dim);    
        let slice = py_to_slice(slice)?;
        return impl_varient_method!(self, t, t.slice(dim, &slice).map_err(to_value_error).map(Into::into));
    }

    fn reshape(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        return impl_varient_method!(self, t, t.reshape(shape).map_err(to_value_error).map(Into::into));
    }

    fn transpose(&self, dim1: isize, dim2: isize) -> PyResult<Self> {
        let dim1 = py_to_dim(dim1);
        let dim2 = py_to_dim(dim2);
        return impl_varient_method!(self, t, t.transpose(dim1, dim2).map_err(to_value_error).map(Into::into));
    }

    fn permute(&self, dims: Vec<usize>) -> PyResult<Self> {
        return impl_varient_method!(self, t, t.permute(dims).map_err(to_value_error).map(Into::into));
    }

    #[staticmethod]
    fn cat(tensors: Vec<Self>, dim: isize) -> PyResult<Self> {
        let dim = py_to_dim(dim);
        if tensors.is_empty() {
            Err(PyValueError::new_err("empty cat"))
        } else {
            let a0 = &tensors[0];
            return impl_varient_method!(
                a0, 
                a0, 
                {
                    let mut vec = vec![a0.clone()];
                    for a in tensors.iter().skip(1) {
                        let t = a.inner.as_tensor().map_err(to_value_error)?;
                        vec.push(t);   
                    }
                    Tensor::cat(&vec, dim).map_err(to_value_error).map(Into::into)
                }
            );
        }
    }

    #[staticmethod]
    fn stack(tensors: Vec<Self>, dim: isize) -> PyResult<Self> {
        let unsqueezed_tensors: PyResult<Vec<_>> = tensors.into_iter()
            .map(|t| t.unsqueeze(dim))
            .collect(); 
        Self::cat(unsqueezed_tensors?, dim)
    }

    fn split(&self, dim: isize) -> PyResult<Vec<Self>> {
        let dim = py_to_dim(dim);
        return impl_varient_method!(self, t, t.split(dim).map_err(to_value_error).map(|v| v.into_iter().map(Into::into).collect()));
    }

    fn chunk(&self, chunks: usize, dim: isize) -> PyResult<Vec<Self>> {
        let dim = py_to_dim(dim);
        return impl_varient_method!(self, t, t.chunk(chunks, dim).map_err(to_value_error).map(|v| v.into_iter().map(Into::into).collect()));
    }

    fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let start_dim = py_to_dim(start_dim);
        let end_dim = py_to_dim(end_dim);
        impl_varient_method!(self, t, t.flatten(start_dim, end_dim).map_err(to_value_error).map(Into::into))
    }

    fn flatten_all(&self) -> PyResult<Self> {
        impl_varient_method!(self, t, t.flatten_all().map_err(to_value_error).map(Into::into))
    }

    fn repeat_dim(&self, dim: isize, times: usize) -> PyResult<Self> {
        let dim = py_to_dim(dim);
        impl_varient_method!(self, t, t.repeat_dim(dim, times).map_err(to_value_error).map(Into::into))
    }

    fn broadcast_as(&self, shape: &Bound<'_, PyAny>) -> PyResult<Self> {
        let shape = py_to_shape(shape)?;
        impl_varient_method!(self, t, t.broadcast_as(shape).map_err(to_value_error).map(Into::into))
    }

    fn if_else(&self, true_val: &Bound<'_, PyAny>, false_val: &Bound<'_, PyAny>) -> PyResult<Self> {
        let confition_val = self.to_bool();
        match (true_val.extract::<PyTensor>(), false_val.extract::<PyTensor>()) {
            (Ok(true_val), Ok(false_val)) => {
                match (&true_val.inner, &false_val.inner) {
                    (DynTensor::F32(true_val), DynTensor::F32(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::F64(true_val), DynTensor::F64(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U32(true_val), DynTensor::U32(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::I32(true_val), DynTensor::I32(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::U8(true_val), DynTensor::U8(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    (DynTensor::Bool(true_val), DynTensor::Bool(false_val)) => confition_val.if_else(true_val, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    _ => Err(PyValueError::new_err(format!("true_val dtype {:?} != false_val dtype {:?} dtype", true_val.dtype(), false_val.dtype()))),
                }
            }
            (Ok(true_val), Err(_)) => {
                match &true_val.inner {
                    DynTensor::F32(true_val) => confition_val.if_else(true_val, false_val.extract::<f32>()?).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::F64(true_val) => confition_val.if_else(true_val, false_val.extract::<f64>()?).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::U8(true_val) => confition_val.if_else(true_val, false_val.extract::<u8>()?).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::U32(true_val) => confition_val.if_else(true_val, false_val.extract::<u32>()?).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::I32(true_val) => confition_val.if_else(true_val, false_val.extract::<i32>()?).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::Bool(true_val) => confition_val.if_else(true_val, false_val.extract::<bool>()?).map_err(to_value_error).map(Into::<PyTensor>::into),                        
                }
            }
            (Err(_), Ok(false_val)) => {
                match &false_val.inner {
                    DynTensor::F32(false_val) => confition_val.if_else(true_val.extract::<f32>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::F64(false_val) => confition_val.if_else(true_val.extract::<f64>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::U8(false_val) => confition_val.if_else(true_val.extract::<u8>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::U32(false_val) => confition_val.if_else(true_val.extract::<u32>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::I32(false_val) => confition_val.if_else(true_val.extract::<i32>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),
                    DynTensor::Bool(false_val) => confition_val.if_else(true_val.extract::<bool>()?, false_val).map_err(to_value_error).map(Into::<PyTensor>::into),                        
                }
            }
            (Err(_), Err(_)) => {
                if true_val.is_instance_of::<pyo3::types::PyFloat>() || false_val.is_instance_of::<pyo3::types::PyFloat>() {
                    let t = true_val.extract::<f64>()?;
                    let f = false_val.extract::<f64>()?;
                    confition_val.if_else(t, f).map_err(to_value_error).map(Into::<PyTensor>::into)
                }

                else if true_val.is_instance_of::<pyo3::types::PyBool>() && false_val.is_instance_of::<pyo3::types::PyBool>() {
                    let t = true_val.extract::<bool>()?;
                    let f = false_val.extract::<bool>()?;
                    confition_val.if_else(t, f).map_err(to_value_error).map(Into::<PyTensor>::into)
                }

                else {
                    let t = true_val.extract::<i32>()?;
                    let f = false_val.extract::<i32>()?;
                    confition_val.if_else(t, f).map_err(to_value_error).map(Into::<PyTensor>::into)
                }
            }
        }
    }

    fn true_count(&self) -> PyResult<usize> {
        let bool_tensor = self.to_bool();
        Ok(bool_tensor.true_count())
    }

    fn false_count(&self) -> PyResult<usize> {
        let bool_tensor = self.to_bool();
        Ok(bool_tensor.false_count())
    }

    fn masked_fill(self_: Bound<'_, Self>, mask: &Self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        mask.if_else(value, self_.as_any())
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

    fn __eq__(&self, other: &Self) -> PyResult<bool> {
        self.allclose(other, Some(0.0), Some(0.0))
    }

    fn __str__(&self) -> String {
        impl_varient_method!(self, t, format!("{}", t))
    }

    fn __repr__(&self) -> String {
        impl_varient_method!(self, t, format!("Tensor({}, shape={:?}, dtype={:?})", t, t.shape(), t.dtype()))
    }
}

macro_rules! impl_new_scaler {
    ($to_tensor:ident, $t:ty) => {
        if let Ok(v) = $to_tensor.extract::<$t>() {
            return Tensor::new(v).map_err(to_value_error).map(Into::into)
        } 
    };
}

macro_rules! impl_new_1d {
    ($to_tensor:ident, $t:ty) => {
        if let Ok(v) = $to_tensor.extract::<Vec<$t>>() {
            return Tensor::new(v).map_err(to_value_error).map(Into::into);
        }
    }
}

macro_rules! impl_new_2d {
    ($to_tensor:ident, $t:ty) => {
        if let Ok(v) = $to_tensor.extract::<Vec<Vec<$t>>>() {
            let n = v.len();
            let m = v.first().map(|x| x.len()).unwrap_or(0);
            let flat: Vec<$t> = v.into_iter().flatten().collect();
            let t = Tensor::new(flat).map_err(to_value_error)?.reshape((n, m)).map_err(to_value_error)?;
            return Ok(t.into())
        }
    };
}

macro_rules! impl_new_3d {
    ($to_tensor:ident, $t:ty) => {
        if let Ok(v) = $to_tensor.extract::<Vec<Vec<Vec<$t>>>>() {
            let n1 = v.len();
            let n2 = v.first().map(|x| x.len()).unwrap_or(0);
            let n3 = v.first().and_then(|x| x.first()).map(|x| x.len()).unwrap_or(0);
            let flat: Vec<$t> = v.into_iter().flatten().flatten().collect();
            return Tensor::new(flat)
                .map_err(to_value_error)?
                .reshape((n1, n2, n3))
                .map_err(to_value_error)
                .map(Into::into)
        }
    }
}

impl PyTensor {
    fn to_bool(&self) -> Tensor<bool> {
        match &self.inner {
            DynTensor::Bool(t) => t.clone(),
            DynTensor::F32(t) => t.cast::<bool>(),
            DynTensor::F64(t) => t.cast::<bool>(),
            DynTensor::U32(t) => t.cast::<bool>(),
            DynTensor::U8(t) => t.cast::<bool>(),
            DynTensor::I32(t) => t.cast::<bool>(),
        }
    } 

    fn new_impl(to_tensor: &Bound<'_, PyAny>) -> PyResult<Self> {
        // ===== scalar =====
        impl_new_scaler!(to_tensor, bool);
        impl_new_scaler!(to_tensor, i32);
        impl_new_scaler!(to_tensor, f32);

        // ===== 1D =====
        impl_new_1d!(to_tensor, bool);
        impl_new_1d!(to_tensor, i32);
        impl_new_1d!(to_tensor, f32);

        // ===== 2D =====
        impl_new_2d!(to_tensor, bool);
        impl_new_2d!(to_tensor, i32);
        impl_new_2d!(to_tensor, f32);

        // ===== 3D =====
        impl_new_3d!(to_tensor, bool);
        impl_new_3d!(to_tensor, i32);
        impl_new_3d!(to_tensor, f32);

        Err(PyTypeError::new_err("Unsupported type for Tensor initialization",))
    }
}

fn to_dtype<T>(tensor: &Tensor<T>, dtype: PyDType) -> PyTensor 
where 
    T:  WithDType
      + DTypeConvert<bool>
      + DTypeConvert<f32>
      + DTypeConvert<f64>
      + DTypeConvert<u32>
      + DTypeConvert<i32>
      + DTypeConvert<u8>

{
    if T::DTYPE == dtype.to_dtype() {
        return PyTensor::from(tensor.clone())
    }
    match dtype {
        PyDType::Bool => tensor.cast::<bool>().into(),
        PyDType::Float32 => tensor.cast::<f32>().into(),
        PyDType::Float64 => tensor.cast::<f64>().into(),
        PyDType::UInt32 => tensor.cast::<u32>().into(),
        PyDType::Int32 => tensor.cast::<i32>().into(),
        PyDType::UInt8 => tensor.cast::<u8>().into(),
    }
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

fn py_to_shape(obj: &Bound<'_, PyAny>) -> PyResult<Shape> {
    if obj.is_none() {
        Ok(().into())
    } else if let Ok(dim) = obj.extract::<usize>() {
        Ok(dim.into())
    } else if let Ok(vec) = obj.extract::<Vec<usize>>() {
        Ok(vec.into())
    } else {
        Err(PyTypeError::new_err(format!("Unsupported shape type: {}", obj.get_type())))
    }
}

fn py_to_indexer(obj: &Bound<'_, PyAny>) -> PyResult<Indexer> {
    if let Ok(idx) = obj.extract::<usize>() {
        return Ok(Indexer::Select(idx));
    }
    if let Ok(slice) = py_to_slice(obj) {
        return Ok(Indexer::Slice(slice))
    }
    
    Err(PyTypeError::new_err(format!("Unsupported index type: {}", obj.get_type())))
}

fn py_to_slice(obj: &Bound<'_, PyAny>) -> PyResult<Slice> {
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

        return Ok(Slice {
            start,
            end: Some(indices.stop),
            step,
        });
    }
    Err(PyTypeError::new_err(format!("Unsupported slice type: {}", obj.get_type())))
}

fn py_to_dim(dim: isize) -> D {
    if dim >= 0 {
        D::Index(dim as usize)
    } else {
        D::Minus((-dim) as usize)
    }
}

impl<T: WithDType> From<Tensor<T>> for PyTensor {
    fn from(value: Tensor<T>) -> PyTensor {
        PyTensor { inner: T::into_dyn(value) }
    }
} 


fn to_value_error(e: lumen_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[allow(unused)]
fn to_runtime_error(e: lumen_core::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}