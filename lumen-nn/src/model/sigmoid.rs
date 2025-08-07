use lumen_core::Tensor;
use super::Model;
use anyhow::Result;

pub struct Sigmoid();

impl Model for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.sigmoid())
    }   

    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        [].into_iter()
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self()
    }
}