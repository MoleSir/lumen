use super::Criterion;
use lumen_core::{op, Tensor};
use anyhow::{Context, Result};

pub struct MSELoss();

impl MSELoss {
    pub fn new() -> Self {
        Self()
    }
}

impl Criterion for MSELoss {
    fn loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor> {
        // (predicted - target)^2
        Ok( op::sub(predicted, target).with_context(|| "when predicted - target")?.pow(2.) )
    }
}