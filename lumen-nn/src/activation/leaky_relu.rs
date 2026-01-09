use lumen_core::{FloatDType, Tensor};
use crate::modules::Module;

pub struct LeakyRelu<T: FloatDType> {
    negative_slope: T,
}

impl<T: FloatDType> LeakyRelu<T> {
    pub fn new(negative_slope: T) -> Self {
        Self { negative_slope }
    }

    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok(input.leaky_relu(self.negative_slope))
    }
}

impl<T: FloatDType> Module<T> for LeakyRelu<T> {
}