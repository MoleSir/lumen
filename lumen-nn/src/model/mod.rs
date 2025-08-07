mod linear;
mod sigmoid;
mod mlp;
mod rnn;
mod gcn;
mod softmax;
mod argmax;
mod logsoftmax;

pub use linear::Linear;
pub use sigmoid::Sigmoid;
pub use mlp::MLP;
pub use gcn::GCNConv;
pub use rnn::RNN;
pub use softmax::SoftMax;
pub use logsoftmax::LogSoftMax;
pub use argmax::ArgMax;

use lumen_core::Tensor;
use anyhow::Result;

pub trait Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> impl Iterator<Item = &Tensor>;
}
