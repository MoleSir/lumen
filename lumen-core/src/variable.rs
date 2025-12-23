use std::marker::PhantomData;
use crate::tensor::ToTensor;
use crate::{FloatDType, Result, Shape, Tensor};

pub struct Var<T: FloatDType> {
    data: PhantomData<T>,
}

impl<T: FloatDType> Var<T> {
    #[inline]
    pub fn new<A: ToTensor<T>>(array: A) -> Result<Tensor<T>> {
        Tensor::new_var(array)
    }

    #[inline]
    pub fn full<S: Into<Shape>>(shape: S, value: T) -> Result<Tensor<T>> {
        Tensor::full_var(shape, value)
    }

    #[inline]
    pub fn zeros<S: Into<Shape>>(shape: S) -> Result<Tensor<T>> {
        Tensor::zeros_var(shape)
    }

    #[inline]
    pub fn ones<S: Into<Shape>>(shape: S) -> Result<Tensor<T>> {
        Tensor::ones_var(shape)
    }

    #[inline]
    pub fn arange(start: T, end: T) -> Result<Tensor<T>> {
        Tensor::arange_var(start, end)
    }

    #[inline]
    pub fn from_vec<V: Into<Vec<T>>, S: Into<Shape>>(vec: V, shape: S) -> Result<Tensor<T>> {
        Tensor::from_vec_var(vec, shape)
    }   

    #[inline]
    pub fn eye(size: usize) -> Result<Tensor<T>> {
        Tensor::eye_var(size)
    }

    #[inline]
    pub fn tril(size: usize, diagonal: bool) -> Result<Tensor<T>> {
        Tensor::tril_var(size, diagonal)
    }

    #[inline]
    pub fn triu(size: usize, diagonal: bool) -> Result<Tensor<T>> {
        Tensor::triu_var(size, diagonal)
    }

    #[inline]
    pub fn diag(diag: &[T]) -> Result<Tensor<T>> {
        Tensor::diag_var(diag)
    }

    #[inline]
    pub fn linspace(start: T, stop: T, num: usize) -> Result<Tensor<T>> {
        Tensor::linspace_var(start, stop, num)
    }

    #[inline]
    pub fn randn<S: Into<Shape>>(mean: T, std: T, shape: S) -> Result<Tensor<T>> {
        Tensor::randn_var(mean, std, shape)
    }

    #[inline]
    pub fn rand<S: Into<Shape>>(min: T, max: T, shape: S) -> Result<Tensor<T>> {
        Tensor::rand_var(min, max, shape)
    }
}