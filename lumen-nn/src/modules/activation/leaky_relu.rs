use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;

#[derive(Module)]
#[module(display = "display")]
pub struct LeakyRelu<T: FloatDType> {
    #[module(skip)]
    negative_slope: T,
}

impl<T: FloatDType> LeakyRelu<T> {
    pub fn new(negative_slope: T) -> Self {
        Self { negative_slope }
    }

    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.leaky_relu(self.negative_slope)?)
    }

    pub fn display(&self) -> String {
        format!("negative_slope={}", self.negative_slope)
    }
}

