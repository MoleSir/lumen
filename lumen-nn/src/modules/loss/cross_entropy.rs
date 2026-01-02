use lumen_core::{FloatDType, IntTensor, NumDType, Tensor};
use crate::Module;

pub struct CrossEntropy;

impl CrossEntropy {
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
        crate::functional::cross_entropy_indices(input, target)
    }
}

impl<T: NumDType> Module<T> for CrossEntropy {
}