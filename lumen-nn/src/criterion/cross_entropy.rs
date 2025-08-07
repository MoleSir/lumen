use lumen_core::Tensor;
use super::Criterion;
use anyhow::Result;

pub struct CrossEntropyLoss();

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self()
    }
}

impl Criterion for CrossEntropyLoss {
    fn loss(&self, _predicted: &Tensor, _target: &Tensor) -> Result<Tensor> {
        todo!()
    }
}