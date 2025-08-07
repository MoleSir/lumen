use super::Criterion;
use lumen_core::{Tensor, op};
use anyhow::{Context, Result};


pub struct L1Loss();

impl L1Loss {
    pub fn new() -> Self {
        Self()
    }
}

impl Criterion for L1Loss {
    fn loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor> {
        // abs(predicted - target)
        Ok( op::sub(predicted, target).with_context(|| "when predicted - target")?.abs() )
    }
}