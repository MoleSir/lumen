use lumen_core::{FloatDType, NumDType, Tensor};
use crate::{functional as F, modules::Module};

/// Applies the Softmax function to an n-dimensional input Tensor
/// rescaling them so that the elements of the n-dimensional output Tensor
/// lie in the range [0,1] and sum to 1.
/// 
/// Softmax is defined as:
/// 
/// .. math::
///     \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
///    
pub struct Softmax {
    dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward<T: FloatDType>(&self, xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        F::softmax(xs, self.dim)
    }
}

impl<T: NumDType> Module<T> for Softmax {
}