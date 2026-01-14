use lumen_core::FloatDType;
use lumen_macros::Module;

use crate::{Buffer, Parameter};

#[derive(Module)]
pub struct BatchNorm<T: FloatDType> {
    pub gamma: Parameter<T>,
    pub beta: Parameter<T>,

    pub running_mean: Buffer<T>,
    pub running_var: Buffer<T>,

    #[module(skip)]
    pub num_features: usize,
    #[module(skip)]
    pub eps: T,
    #[module(skip)]
    pub momentum: T,
}