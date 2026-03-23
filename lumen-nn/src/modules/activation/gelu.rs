use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{NnResult, NnCtxError, ModuleForward};

#[derive(Module, Clone)]
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        Ok(input.gelu()?)
    }
}

impl<T: FloatDType> ModuleForward<T> for Gelu {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        Gelu::forward(self, &input)
    }
}