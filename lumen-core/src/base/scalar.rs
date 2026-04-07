use crate::{Tensor, WithDType};

pub enum TensorOrScalar<T: WithDType> {
    Tensor(Tensor<T>),
    Scalar(T),
}

impl<T: WithDType> From<T> for TensorOrScalar<T> {
    fn from(value: T) -> Self {
        TensorOrScalar::Scalar(value)
    }
}

impl<T: WithDType> From<Tensor<T>> for TensorOrScalar<T> {
    fn from(value: Tensor<T>) -> Self {
        TensorOrScalar::Tensor(value)
    }
}

impl<T: WithDType> From<&Tensor<T>> for TensorOrScalar<T> {
    fn from(value: &Tensor<T>) -> Self {
        TensorOrScalar::Tensor(value.clone())
    }
}

