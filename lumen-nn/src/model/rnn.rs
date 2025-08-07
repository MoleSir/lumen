use lumen_core::{op, Tensor, TensorError, rng, rngs, Range};
use crate::init;
use anyhow::{Context, Result};

pub struct RNN {
    input_size: usize,
    hidden_size: usize,

    weight_ih: Tensor,
    weight_hh: Tensor,
    bias: Tensor,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            
            weight_ih: init::kaiming_default_uniform([input_size, hidden_size]).require_grad(),
            weight_hh: init::kaiming_default_uniform([hidden_size, hidden_size]).require_grad(),
            bias: init::zero([hidden_size]).require_grad(),
        }
    }

    pub fn forward(&self, x: &Tensor, h0: Option<&Tensor>) -> Result<Tensor> {
        // x: (batch_size, seq_length, input_size)
        // h0: (batch_size, hidden_size)
        if x.dim_size() != 3 {
            return Err(TensorError::DimensionsUnmatch).with_context(|| "Check input tensor dim");
        }
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let input_size = x.shape()[2];
        if input_size != self.input_size {
            return Err(TensorError::DifferentShape).with_context(|| "Check input tensor size");
        }

        let h0 = match h0 {
            Some(h0) => h0.clone(),
            None => Tensor::zeros([batch_size, self.hidden_size])
        };

        if *h0.shape() != [batch_size, self.hidden_size] {
            return Err(TensorError::DifferentShape).with_context(|| "Check hidden tensor size");
        }

        let mut outputs = Vec::new();

        let mut h_t = h0.clone();
        for t in 0..seq_length {
            // x_t: (batch_size, input_size)
            // h_t: (batch_size, hidden_size)
            let x_t = x.slice(rngs!((:), (t), (:))).unwrap();
            h_t = op::tanh(&(
                op::matmul(&x_t, &self.weight_ih).with_context(|| "@ with `weight ih`")? +
                op::matmul(&h_t, &self.weight_hh).with_context(|| "@ with `weight hh`")? +
                &self.bias
            ));

            outputs.push(h_t.clone());            
        }
        
        // (seq_len, batch_size, hidden_size)
        // TODO: stack in dim
        Ok(op::stack(&outputs)?)
    }

    pub fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        [&self.weight_ih, &self.weight_hh, &self.bias].into_iter()
    }
}

#[allow(unused)]
#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_sin_to_cos() {
        let rnn = RNN::new(1, 8);
        let x = Tensor::zeros([5, 10, 1]);
        let h0 = Tensor::zeros([5, 8]);
        let hs = rnn.forward(&x, Some(&h0)).unwrap();
        hs.backward();
    }
}