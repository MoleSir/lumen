
use crate::Tensor;

#[derive(Clone)]
pub enum FloatTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
}

impl From<Tensor<f32>> for FloatTensor {
    fn from(value: Tensor<f32>) -> Self {
        Self::F32(value)
    }
}

impl From<&Tensor<f32>> for FloatTensor {
    fn from(value: &Tensor<f32>) -> Self {
        Self::F32(value.clone())
    }
}

impl From<Tensor<f64>> for FloatTensor {
    fn from(value: Tensor<f64>) -> Self {
        Self::F64(value)
    }
}

impl From<&Tensor<f64>> for FloatTensor {
    fn from(value: &Tensor<f64>) -> Self {
        Self::F64(value.clone())
    }
}