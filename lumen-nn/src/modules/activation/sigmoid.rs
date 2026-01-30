use lumen_core::{FloatDType, Tensor};
use crate::Module;

#[derive(Module)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.sigmoid()?)
    }
}
