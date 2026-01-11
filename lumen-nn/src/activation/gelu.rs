use lumen_core::{FloatDType, Tensor};
use crate::modules::Module;

pub struct Gelu;

impl Gelu {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.gelu())
    }
}

impl<T: FloatDType> Module<T> for Gelu {
}