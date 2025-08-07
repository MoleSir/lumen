use lumen_core::{op, Tensor};
use anyhow::{Context, Result};

pub struct GCNConv {
    weights: Tensor,
    bias: Tensor,
}

impl GCNConv {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::rand([in_features, out_features]),
            bias: Tensor::rand([out_features])
        }
    }

    pub fn forward(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        let out = op::matmul(adj, x).with_context(|| "When `adj` @ `x`")?;
        let out = op::matmul(&out, &self.weights).with_context(|| "When @ with `weights`")?;
        op::add(&out, &self.bias).with_context(|| "When add with `bias`")
    }
}

