use lumen_core::{FloatDType, IntTensor, Tensor};
use lumen_macros::Module;
use crate::{functional::LossReduction, NnResult};

#[derive(Module, Default)]
#[module(display = "display")]
pub struct CrossEntropyLoss {
    #[module(skip)]
    reduction: LossReduction,
}

impl CrossEntropyLoss {
    pub fn new(reduction: LossReduction) -> Self {
        Self {reduction }
    }
    
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
        crate::functional::cross_entropy(input, target, self.reduction)
    }

    fn display(&self) -> String {
        format!("reduction={}", self.reduction.to_str())
    }
}
