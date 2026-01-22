use lumen_nn::init::Init;
use pyo3::{exceptions::PyValueError, prelude::*};
use crate::core::{py_to_shape, to_value_error, PyDType, PyTensor};

// ============================================================================= //
//                         PyO3 Wrapper
// ============================================================================= //

#[pyclass(name = "Init")] 
#[derive(Clone, Debug)]
pub struct PyInit {
    pub inner: Init<f64>,
}

#[pymethods]
impl PyInit {
    #[staticmethod]
    pub fn uninit() -> Self {
        Self { inner: Init::Uninit }
    }

    #[staticmethod]
    pub fn empty() -> Self {
        Self { inner: Init::Empty }
    }

    #[staticmethod]
    pub fn ones() -> Self {
        Self { inner: Init::Ones }
    }

    #[staticmethod]
    pub fn zeros() -> Self {
        Self { inner: Init::Zeros }
    }

    #[staticmethod]
    #[pyo3(signature = (value))]
    pub fn constant(value: f64) -> Self {
        Self { inner: Init::Constant { value } }
    }

    #[staticmethod]
    #[pyo3(signature = (min=0.0, max=1.0))]
    pub fn uniform(min: f64, max: f64) -> Self {
        Self { inner: Init::Uniform { min, max } }
    }

    #[staticmethod]
    #[pyo3(signature = (mean=0.0, std=1.0))]
    pub fn normal(mean: f64, std: f64) -> Self {
        Self { inner: Init::Normal { mean, std } }
    }

    #[staticmethod]
    #[pyo3(signature = (gain=1.0, fan_out_only=false))] 
    pub fn kaiming_uniform(gain: f64, fan_out_only: bool) -> Self {
        Self { inner: Init::KaimingUniform { gain, fan_out_only } }
    }

    #[staticmethod]
    #[pyo3(signature = (gain=1.0, fan_out_only=false))]
    pub fn kaiming_normal(gain: f64, fan_out_only: bool) -> Self {
        Self { inner: Init::KaimingNormal { gain, fan_out_only } }
    }

    #[staticmethod]
    #[pyo3(signature = (gain=1.0))]
    pub fn xavier_uniform(gain: f64) -> Self {
        Self { inner: Init::XavierUniform { gain } }
    }

    #[staticmethod]
    #[pyo3(signature = (gain=1.0))]
    fn xavier_normal(gain: f64) -> Self {
        Self { inner: Init::XavierNormal { gain } }
    }

    #[pyo3(signature = (shape, dtype, fan_in=None, fan_out=None))]
    pub fn init(&self, shape: &Bound<'_, PyAny>, dtype: PyDType, fan_in: Option<usize>, fan_out: Option<usize>) -> PyResult<PyTensor> {
        let shape = py_to_shape(shape)?;
        match dtype {
            PyDType::Float32 => self.to_f32().do_init_with(shape, fan_in, fan_out).map_err(to_value_error).map(Into::into),
            PyDType::Float64 => self.to_f64().do_init_with(shape, fan_in, fan_out).map_err(to_value_error).map(Into::into),
            _ => Err(PyValueError::new_err(format!("init unsupport dtype {:?}", dtype)))
        }
    }
    
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

impl PyInit {
    pub fn to_f32(&self) -> Init<f32> {
        match self.inner {
            Init::Uninit => Init::Uninit,
            Init::Empty => Init::Empty,
            Init::Constant { value } => Init::Constant { value: value as f32 },
            Init::Ones => Init::Ones,
            Init::Zeros => Init::Zeros,
            Init::Uniform { min, max } => Init::Uniform { min: min as f32, max: max as f32 },
            Init::Normal { mean, std } => Init::Normal { mean: mean as f32, std: std as f32 },
            Init::KaimingUniform { gain, fan_out_only } => Init::KaimingUniform { gain: gain as f32, fan_out_only },
            Init::KaimingNormal { gain, fan_out_only } => Init::KaimingNormal { gain: gain as f32, fan_out_only },
            Init::XavierUniform { gain } => Init::XavierUniform { gain: gain as f32 },
            Init::XavierNormal { gain } => Init::XavierNormal { gain: gain as f32 },
        }
    }

    pub fn to_f64(&self) -> Init<f64> {
        self.inner
    }
}