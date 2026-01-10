use std::marker::PhantomData;
use lumen_core::{Tensor, WithDType};

pub trait Batcher {
    type Item;
    type Output;
    type Error;
    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error>;
}

#[derive(Default)]
pub struct TensorPairBatcher<T>(PhantomData<T>);

impl<T: WithDType> Batcher for TensorPairBatcher<T> {
    type Error = lumen_core::Error;
    type Item = (Tensor<T>, Tensor<T>);
    type Output = (Tensor<T>, Tensor<T>);

    fn batch(&self, items: Vec<(Tensor<T>, Tensor<T>)>) -> Result<Self::Output, Self::Error> {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?; 
        Ok((xs, ys))
    }
}