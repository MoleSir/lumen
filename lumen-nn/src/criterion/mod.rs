mod l1;
mod mse;
mod cross_entropy;

pub use mse::MSELoss;
pub use l1::L1Loss;
pub use cross_entropy::CrossEntropyLoss;

use lumen_core::Tensor;
use anyhow::Result;

pub trait Criterion {
    fn loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor>;
}