use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;

#[derive(Module)]
pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.relu()?)
    }
}

