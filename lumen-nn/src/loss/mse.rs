use lumen_core::{FloatDType, Tensor};
use crate::{Module, NnResult};

pub struct MseLoss;

impl MseLoss {
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: &Tensor<T>) -> NnResult<Tensor<T>> {
        crate::functional::mse_loss(input, target)
    }
}

impl<T: FloatDType> Module<T> for MseLoss {
}