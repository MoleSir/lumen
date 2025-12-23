use lumen_core::{FloatDType, NumDType, Tensor};
use crate::modules::Module;

pub struct Gelu;

impl Gelu {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: Tensor<T>) -> Tensor<T> {
        input.gelu()
    }
}

impl<T: NumDType> Module<T> for Gelu {
}