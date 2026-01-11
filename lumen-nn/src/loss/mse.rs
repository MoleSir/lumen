use lumen_core::{FloatDType, Tensor};
use crate::Module;

pub struct MseLoss;

impl MseLoss {
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        crate::functional::mse_loss(input, target)
    }
}

impl<T: FloatDType> Module<T> for MseLoss {
}