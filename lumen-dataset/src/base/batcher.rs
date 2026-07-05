use std::{convert::Infallible, fmt::Display, marker::PhantomData};
use lumen_core::{Tensor, WithDType};

pub trait Batcher {
    type Item;
    type Output;
    type Error: Display;

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error>;
}

#[derive(Default)]
pub struct TensorPairBatcher<T1, T2>(PhantomData<T1>, PhantomData<T2>);

impl<T1: WithDType, T2: WithDType> Batcher for TensorPairBatcher<T1, T2> {
    type Item = (Tensor<T1>, Tensor<T2>);
    type Output = (Tensor<T1>, Tensor<T2>);
    type Error = lumen_core::Error;

    fn batch(&self, items: Vec<(Tensor<T1>, Tensor<T2>)>) -> Result<Self::Output, Self::Error> {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?; 
        Ok((xs, ys))
    }
}

#[derive(Default)]
pub struct NoBatcher<T>(PhantomData<T>);

impl<T> Batcher for NoBatcher<T> {
    type Error = Infallible;
    type Item = T;
    type Output = Vec<T>;

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error> {
        Ok(items)
    }
}

#[derive(Default)]
pub struct PairBatcher<T1, T2>((PhantomData<T1>, PhantomData<T2>));

impl<T1, T2> Batcher for PairBatcher<T1, T2> {
    type Error = Infallible;
    type Item = (T1, T2);
    type Output = (Vec<T1>, Vec<T2>);

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error> {
        let (xs, ys): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((xs, ys))
    }
}