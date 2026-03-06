use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{functional as F, ModuleForward, NnCtxError, NnResult};

/// Applies the Softmax function to an n-dimensional input Tensor
/// rescaling them so that the elements of the n-dimensional output Tensor
/// lie in the range [0,1] and sum to 1.
/// 
/// Softmax is defined as:
/// 
/// .. math::
///     \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
///    
#[derive(Module)]
pub struct Softmax {
    #[module(skip)]
    dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward<T: FloatDType>(&self, xs: &Tensor<T>) -> NnResult<Tensor<T>> {
        F::softmax(xs, self.dim)
    }
}


impl<T: FloatDType> ModuleForward<T> for Softmax {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        Softmax::forward(self, &input)   
    }
}