use crate::{Result, Tensor, WithDType};

pub enum TensorScalar<T: WithDType> {
    Tensor(Tensor<T>),
    Scalar(Tensor<T>),
}

pub trait TensorOrScalar<T: WithDType> {
    fn to_tensor_scalar(self) -> Result<TensorScalar<T>>;
}

impl<T: WithDType> TensorOrScalar<T> for &Tensor<T> {
    fn to_tensor_scalar(self) -> Result<TensorScalar<T>> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

impl<T: WithDType> TensorOrScalar<T> for Tensor<T> {
    fn to_tensor_scalar(self) -> Result<TensorScalar<T>> {
        Ok(TensorScalar::Tensor(self.clone()))
    }
}

impl<T: WithDType> TensorOrScalar<T> for T {
    fn to_tensor_scalar(self) -> Result<TensorScalar<T>> {
        let scalar = Tensor::new(self)?;
        Ok(TensorScalar::Scalar(scalar))
    }
}
