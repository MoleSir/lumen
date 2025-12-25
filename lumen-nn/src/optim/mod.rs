mod sgd;
mod adamw;
pub use sgd::*;
pub use adamw::*;

use lumen_core::{FloatDType, GradStore};

pub trait Optimizer<T: FloatDType> {
    type Error: std::error::Error;
    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error>;
}