use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{functional::{reduction_display, LossReduction}, NnResult};

#[derive(Module)]
#[module(display = "display")]
pub struct L1Loss {
    #[module(skip)]
    reduction: Option<LossReduction>,
}

impl L1Loss {
    pub fn new(reduction: Option<LossReduction>) -> Self {
        Self {reduction }
    }
    
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: &Tensor<T>) -> NnResult<Tensor<T>> {
        crate::functional::mse_loss(input, target, self.reduction)
    }

    fn display(&self) -> String {
        reduction_display(self.reduction).to_string()
    }
}
