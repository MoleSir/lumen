use std::marker::PhantomData;
use lumen_core::{Tensor, WithDType};

pub trait Batcher {
    type Item;
    type Output;
    fn batch(&self, items: Vec<Self::Item>) -> Self::Output;
}

#[derive(Default)]
pub struct TensorPairBatcher<T>(PhantomData<T>);

impl<T: WithDType> Batcher for TensorPairBatcher<T> {
    type Item = (Tensor<T>, Tensor<T>);
    type Output = Result<(Tensor<T>, Tensor<T>), lumen_core::Error>;

    fn batch(&self, items: Vec<(Tensor<T>, Tensor<T>)>) -> Self::Output {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?; 
        Ok((xs, ys))
    }
}