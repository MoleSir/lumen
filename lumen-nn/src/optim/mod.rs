mod sgd;
mod momentum;
mod adam;

pub use sgd::{SDG, SDGConfig};
pub use momentum::{Momentum, MomentumConfig};
pub use adam::{Adam, AdamConfig};

use lumen_core::Tensor;

pub trait Optim {
    fn step(&mut self);
    fn parameters(&self) -> impl Iterator<Item = &Tensor>;
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}
