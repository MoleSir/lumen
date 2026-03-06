use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{NnResult, NnCtxError, ModuleForward};

#[derive(Module)]
pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        Ok(input.relu()?)
    }
}

impl<T: FloatDType> ModuleForward<T> for Relu {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        Relu::forward(self, &input)
    }
}