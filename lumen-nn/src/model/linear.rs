use lumen_core::{op, Tensor};
use crate::init;
use super::Model;
use anyhow::{Context, Ok, Result};

pub struct Linear {
    in_features: usize, 
    #[allow(unused)]
    out_features: usize,

    weights: Tensor,
    biases: Option<Tensor>,
}

impl Model for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let y = op::matmul(input, &self.weights).with_context(|| "@ with `weights`")?;
        match &self.biases {
            Some(biases) => op::add(&y, biases).with_context(|| "add with `biases`"),
            None => Ok(y),
        }
    }

    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        match &self.biases {
            Some(biases) => [&self.weights, biases].to_vec(),
            None => [&self.weights].to_vec(),
        }.into_iter()
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // input: (n, i)
        // weights: (i, o)
        // bias: (1, o)
        // output = input @ weights + bias
        let weights = init::kaiming_default_normal([in_features, out_features]).require_grad();
        if bias {
            Self { 
                in_features, out_features,
                weights,
                biases: Some(init::kaiming_default_normal([out_features]).require_grad()),
            }
        } else {
            Self { 
                in_features, out_features,
                weights, biases: None
            }
        }
    }

    pub fn fit(&self, input: &Tensor, target: &Tensor, epochs: usize, learn_rate: f64) -> Result<()> {
        let batch_size = input.element_size() / self.in_features;
        let input = input.reshape([batch_size, self.in_features]).unwrap().require_grad();

        for _ in 0..epochs {
            let pred = self.forward(&input).with_context(|| "forward")?;
            let loss = (&pred - target).pow(2.);
            loss.backward();

            for (v, &g) in self.weights.iter_mut().zip(self.weights.grad().unwrap().iter()) {
                *v += -learn_rate * g;
            }
            self.weights.zero_grad();

            if let Some(biases) = &self.biases {
                for (v, &g) in biases.iter_mut().zip(biases.grad().unwrap().iter()) {
                    *v += -learn_rate * g;
                }
                biases.zero_grad();
            }
        } 

        Ok(())
    }
}

#[cfg(test)]
#[allow(unused)]
mod test {
    use super::*;

    #[test]
    fn test_train_1_batch() {
        const TARGET: f64 = 0.36;
        let l = Linear::new(3, 1, true);

        let input = Tensor::build([
            0.5, 0.3, 0.8
        ], [1, 3]).unwrap().require_grad();

        let target = Tensor::build([
            TARGET,
        ], [1, 1]).unwrap().require_grad();

        l.fit(&input, &target, 100, 0.1);

        let pred = l.forward(&input).unwrap();
        assert!(approx::abs_diff_eq!(pred.to_vec()[0], TARGET));
    }

    #[test]
    fn test_train_mul_batch() {
        let l = Linear::new(3, 1, true);

        let input = Tensor::build([
            0.5, 0.3, 0.8,
            0.4, 0.1, 0.12,
        ], [2, 3]).unwrap().require_grad();

        let target = Tensor::build([
            0.4, 
            0.9,
        ], [2, 1]).unwrap().require_grad();

        l.fit(&input, &target, 500, 0.1);

        let pred = l.forward(&input).unwrap();
        assert!(approx::abs_diff_eq!(pred.to_vec()[0], 0.4, epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[1], 0.9, epsilon=0.01));
    }
}
