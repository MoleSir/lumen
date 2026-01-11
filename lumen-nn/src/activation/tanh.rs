use lumen_core::{FloatDType, Tensor};
use crate::modules::Module;

pub struct Tanh;

impl Tanh {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.tanh())
    }
}

impl<T: FloatDType> Module<T> for Tanh {
}