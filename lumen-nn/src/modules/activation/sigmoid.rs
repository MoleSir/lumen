use lumen_core::{FloatDType, Tensor};
use crate::Module;
use crate::{NnResult, NnCtxError, ModuleForward};

#[derive(Module)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        Ok(input.sigmoid()?)
    }
}

impl<T: FloatDType> ModuleForward<T> for Sigmoid {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        Sigmoid::forward(self, &input)
    }
}