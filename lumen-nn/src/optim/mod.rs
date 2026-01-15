mod sgd;
mod adamw;
mod momentum;
pub use sgd::*;
pub use adamw::*;
pub use momentum::*;

use lumen_core::{FloatDType, GradStore};

pub trait Optimizer<T: FloatDType> {
    type Error: std::error::Error + Sync + Send + 'static;
    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error>;
}