use lumen_core::{FloatDType, GradStore, Tensor};
use super::Optimizer;

#[derive(Debug)]
pub struct SGD<T: FloatDType> {
    pub params: Vec<Tensor<T>>,
    pub learning_rate: T,
}

impl<T: FloatDType> SGD<T> {
    pub fn new(params: impl Into<Vec<Tensor<T>>>, learning_rate: T) -> Self {
        Self { params: params.into(), learning_rate }
    }
}

impl<T: FloatDType> Optimizer<T> for SGD<T> {
    type Error = lumen_core::CtxError;
    fn step(&mut self, grads: &GradStore<T>) -> lumen_core::Result<()> {
        for var in self.params.iter() {
            if let Some(grad) = grads.get(var) {
                var.sub_(self.learning_rate * grad)?;
            }
        }
        Ok(())
    }
}