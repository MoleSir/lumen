use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{functional::LossReduction, NnResult};

#[derive(Module, Default)]
#[module(display = "display")]
pub struct MseLoss {
    #[module(skip)]
    reduction: LossReduction,
}

impl MseLoss {
    pub fn new(reduction: LossReduction) -> Self {
        Self {reduction }
    }
    
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: &Tensor<T>) -> NnResult<Tensor<T>> {
        crate::functional::mse_loss(input, target, self.reduction)
    }

    fn display(&self) -> String {
        format!("reduction={}", self.reduction.to_str())
    }
}
