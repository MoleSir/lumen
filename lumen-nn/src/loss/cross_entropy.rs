use lumen_core::{FloatDType, IntTensor, Tensor};
use crate::{Module, NnResult};

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn forward<T: FloatDType>(&self, input: &Tensor<T>, target: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
        crate::functional::cross_entropy_indices(input, target)
    }
}

impl<T: FloatDType> Module<T> for CrossEntropyLoss {
}