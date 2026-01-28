use lumen_core::DynTensor;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyFloat};
use crate::{core::{py_to_dim, to_value_error, PyTensor}, impl_floatdtype_varient_method, impl_intdtype_varient_method, impl_varient_method};
use lumen_nn::functional::{self as F, LossReduction};

// ============================================================================= //
//                         Common 
// ============================================================================= //

macro_rules! match_linear_with_bias {
    ($input:expr, $weight:expr, $bias:expr; $($t:ident),*) => {
        match (&$input.inner, &$weight.inner, &$bias.inner) {
            $(
                (DynTensor::$t(i), DynTensor::$t(w), DynTensor::$t(b)) => F::linear(i, w, Some(b)).map_err(|e| PyValueError::new_err(e.to_string())).map(Into::into),
            )*
            _ => Err(PyValueError::new_err("unmatch dtype in linear"))
        }
    };
}

macro_rules! match_linear_no_bias {
    ($input:expr, $weight:expr; $($t:ident),*) => {
        match (&$input.inner, &$weight.inner) {
            $(
                (DynTensor::$t(i), DynTensor::$t(w)) => F::linear(i, w, None).map_err(|e| PyValueError::new_err(e.to_string())).map(Into::into),
            )*
            _ => Err(PyValueError::new_err("unmatch dtype in linear"))
        }
    };
}

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None))]
fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    match bias {
        Some(bias) => match_linear_with_bias!(input, weight, bias; F32, F64, U32, I32, U8),
        None => match_linear_no_bias!(input, weight; F32, F64, U32, I32, U8),
    }
}

#[pyfunction]
fn softmax(xs: &PyTensor, dim: isize) -> PyResult<PyTensor> {
    let dim = py_to_dim(dim);
    impl_floatdtype_varient_method!(xs, t, F::softmax(t, dim).map_err(to_value_error).map(Into::into), "softmax")
}

#[pyfunction]
fn log_softmax(xs: &PyTensor, dim: isize) -> PyResult<PyTensor> {
    let dim = py_to_dim(dim);
    impl_floatdtype_varient_method!(xs, t, F::log_softmax(t, dim).map_err(to_value_error).map(Into::into), "log_softmax")
}

#[pyfunction]
fn dropout(xs: &PyTensor, drop_p: &Bound<'_, PyFloat>) -> PyResult<PyTensor> {
    impl_floatdtype_varient_method!(xs, t, F::dropout(t, drop_p.extract()?).map_err(to_value_error).map(Into::into), "dropout")
} 

#[pyfunction]
fn embedding(weight: &PyTensor, indexes: &PyTensor) -> PyResult<PyTensor> {
    impl_varient_method!(
        weight, weight, 
        impl_intdtype_varient_method!(
            indexes, indexes, 
            F::embedding(weight, indexes).map_err(to_value_error).map(Into::into),
            "embedding"
        )
    )
}

// ============================================================================= //
//                         Activate 
// ============================================================================= //

macro_rules! impl_active {
    ($($method:ident),*) => {$(
        #[pyfunction]
        fn $method(xs: &PyTensor) -> PyResult<PyTensor> {
            impl_floatdtype_varient_method!(xs, t, F::$method(t).map_err(to_value_error).map(Into::into), stringify!($method))
        }
    )*};
}

impl_active!(silu, sigmoid, relu, hard_sigmoid);

#[pyfunction]
#[pyo3(signature = (xs, negative_slope=1e-5))]
fn leaky_relu(xs: &PyTensor, negative_slope: f64) -> PyResult<PyTensor> {
    match &xs.inner {
        DynTensor::F32(t) => F::leaky_relu(t, negative_slope as f32).map_err(to_value_error).map(Into::into),
        DynTensor::F64(t) => F::leaky_relu(t, negative_slope).map_err(to_value_error).map(Into::into),
        _ => Err(PyValueError::new_err(format!("{:?} dtype unsupport {}", xs.dtype(), "leaky_relu"))),
    }
}

// ============================================================================= //
//                         Loss 
// ============================================================================= //

#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
fn nll_loss(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    let reduction = py_to_reduction(reduction)?;
    impl_floatdtype_varient_method!(
        input, input, 
        impl_intdtype_varient_method!(target, target, F::nll_loss(input, target, reduction).map_err(to_value_error).map(Into::into), "nll_loss"),
        "nll_loss"
    )
}

#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
fn l1_loss(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    let reduction = py_to_reduction(reduction)?;
    match (&input.inner, &target.inner) {
        (DynTensor::F32(input), DynTensor::F32(target)) => {
            F::l1_loss(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), DynTensor::F64(target)) => {
            F::l1_loss(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        _ => Err(PyValueError::new_err(format!("not float dtype in messloss")))
    }
}

#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
fn mse_loss(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    let reduction = py_to_reduction(reduction)?;
    match (&input.inner, &target.inner) {
        (DynTensor::F32(input), DynTensor::F32(target)) => {
            F::mse_loss(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), DynTensor::F64(target)) => {
            F::mse_loss(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        _ => Err(PyValueError::new_err(format!("not float dtype in messloss")))
    }
}

#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
fn soft_cross_entropy(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    let reduction = py_to_reduction(reduction)?;
    match (&input.inner, &target.inner) {
        (DynTensor::F32(input), DynTensor::F32(target)) => {
            F::soft_cross_entropy(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), DynTensor::F64(target)) => {
            F::soft_cross_entropy(input, target, reduction).map_err(to_value_error).map(Into::into)
        }
        _ => Err(PyValueError::new_err(format!("not float dtype in cross_entropy")))
    }
}

#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
fn cross_entropy(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    let reduction = py_to_reduction(reduction)?;
    impl_floatdtype_varient_method!(
        input, input, 
        impl_intdtype_varient_method!(target, target, F::cross_entropy(input, target, reduction).map_err(to_value_error).map(Into::into), "cross_entropy_indices"),
        "cross_entropy_indices"
    )
}

// ============================================================================= //
//                         Loss 
// ============================================================================= //

#[pyfunction]
#[pyo3(signature = (input, weight=None, bias=None, eps=1e-5))]
fn layer_norm(input: &PyTensor, weight: Option<&PyTensor>, bias: Option<&PyTensor>, eps: f64) -> PyResult<PyTensor> {
    match (&input.inner, weight.map(|w| &w.inner), bias.map(|w| &w.inner)) {
        (DynTensor::F32(input), Some(DynTensor::F32(weight)), Some(DynTensor::F32(bias))) => {
            F::layer_norm(input, Some(weight), Some(bias), eps as f32).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F32(input), Some(DynTensor::F32(weight)), None) => {
            F::layer_norm(input, Some(weight), None, eps as f32).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F32(input), None, Some(DynTensor::F32(bias))) => {
            F::layer_norm(input, None, Some(bias), eps as f32).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F32(input), None, None) => {
            F::layer_norm(input, None, None, eps as f32).map_err(to_value_error).map(Into::into)
        }

        (DynTensor::F64(input), Some(DynTensor::F64(weight)), Some(DynTensor::F64(bias))) => {
            F::layer_norm(input, Some(weight), Some(bias), eps).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), Some(DynTensor::F64(weight)), None) => {
            F::layer_norm(input, Some(weight), None, eps).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), None, Some(DynTensor::F64(bias))) => {
            F::layer_norm(input, None, Some(bias), eps).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), None, None) => {
            F::layer_norm(input, None, None, eps).map_err(to_value_error).map(Into::into)
        }

        _ => Err(PyValueError::new_err("unmatch dtype in layer_norm"))
    }
}

fn py_to_reduction(reduction: &str) -> PyResult<LossReduction> {
    match reduction {
        "none" => Ok(LossReduction::None),
        "sum" => Ok(LossReduction::Sum),
        "mean" => Ok(LossReduction::Mean),
        _ => Err(PyValueError::new_err(format!("unsupport reduction {}", reduction)))
    }
}

#[pyfunction]
#[pyo3(signature = (input, weight, eps=1e-5))]
fn rms_norm(input: &PyTensor, weight: &PyTensor, eps: f64) -> PyResult<PyTensor> {
    match (&input.inner, &weight.inner) {
        (DynTensor::F32(input), DynTensor::F32(weight)) => {
            F::rms_norm(input, weight, eps as f32).map_err(to_value_error).map(Into::into)
        }
        (DynTensor::F64(input), DynTensor::F64(weight)) => {
            F::rms_norm(input, weight, eps).map_err(to_value_error).map(Into::into)
        }
        _ => Err(PyValueError::new_err(format!("not float dtype in rms_norm")))
    }
}

#[pymodule]
pub fn functional(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(dropout, m)?)?;
    m.add_function(wrap_pyfunction!(embedding, m)?)?;

    m.add_function(wrap_pyfunction!(silu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(hard_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(leaky_relu, m)?)?;

    m.add_function(wrap_pyfunction!(nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(l1_loss, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(soft_cross_entropy, m)?)?;

    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;

    Ok(())
}
