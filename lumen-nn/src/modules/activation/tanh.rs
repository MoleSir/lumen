use lumen_core::{FloatDType, NumDType, Tensor};
use crate::modules::Module;

pub struct Tanh;

impl Tanh {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: Tensor<T>) -> Tensor<T> {
        input.tanh()
    }
}

impl<T: NumDType> Module<T> for Tanh {
}