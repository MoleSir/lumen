
use lumen_core::Tensor;
use anyhow::{bail, Context, Result};

use super::{Linear, Model, Sigmoid};

pub struct MLP {
    archs: Vec<usize>,
    linears: Vec<Linear>,
    activation: Sigmoid,
}

impl Model for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut xs = input.clone();
        for (i, linear) in self.linears.iter().take(self.layer_size() - 1).enumerate() {
            xs = linear.forward(&xs).with_context(|| format!("forward in '{}' linear layer", i))?;
            xs = self.activation.forward(&xs).with_context(|| format!("forward in '{}' activation layer", i))?;
        }
        self.linears.last().unwrap().forward(&xs).with_context(|| "forward in last linear")
    }

    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        self.linears.iter().flat_map(|neural| neural.parameters())
    }
}

impl MLP {
    pub fn from_archs<A: Into<Vec<usize>>>(archs: A) -> Result<Self> {
        let archs = archs.into();
        if archs.len() == 1 {
            bail!("archs size '{}' too small!", archs.len())
        } else {
            let linears = archs.iter().zip(archs.iter().skip(1))
            .map(|(&input_size, &output_size)| -> Linear {
                Linear::new(input_size, output_size, true)
            })
            .collect();
        
            Ok(Self { archs, linears, activation: Sigmoid::new() })
        }
    }

    pub fn input_size(&self) -> usize {
        self.archs[0]
    }

    pub fn output_size(&self) -> usize {
        self.archs.last().unwrap().clone()
    }

    pub fn layer_size(&self) -> usize {
        self.archs.len() - 1
    }
}

#[allow(unused)]
#[cfg(test)]
mod test {
    use crate::criterion::L1Loss;
    use crate::optim::{Momentum, SDGConfig};
    use crate::{criterion::MSELoss, optim::SDG};
    use crate::{criterion::Criterion, optim::Optim};

    use super::*;
    
    #[test]
    fn test_train_and_gate() {
        let input = Tensor::build([
            0., 0.,
            0., 1.,
            1., 0.,
            1., 1.,
        ], [4, 2]).unwrap().require_grad();

        let target = Tensor::build([
            0.,
            0.,
            0.,
            1.,
        ], [4, 1]).unwrap().require_grad();

        let mlp = MLP::from_archs([2, 4, 1]).unwrap();
        let mut optimizer = SDG::with_config(mlp.parameters(), SDGConfig::new(0.1));
        let criterion = MSELoss::new();
    
        for _ in 0..2000 {
            optimizer.zero_grad();
            let pred = mlp.forward(&input).unwrap();
            let loss = criterion.loss(&pred, &target).unwrap();
            loss.backward();
            optimizer.step();
        }

        let pred = mlp.forward(&input).unwrap();
        assert!(approx::abs_diff_eq!(pred.to_vec()[0], 0., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[1], 0., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[2], 0., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[3], 1., epsilon=0.01));
    }

    #[test]
    fn test_train_or_gate() {
        let input = Tensor::build([
            0., 0.,
            0., 1.,
            1., 0.,
            1., 1.,
        ], [4, 2]).unwrap().require_grad();

        let target = Tensor::build([
            0.,
            1.,
            1.,
            1.,
        ], [4, 1]).unwrap().require_grad();

        let mlp = MLP::from_archs([2, 4, 1].to_vec()).unwrap();
        let mut optimizer = SDG::new(mlp.parameters(), 0.1);
        let criterion = MSELoss::new();
    
        for _ in 0..2000 {
            optimizer.zero_grad();
            let pred = mlp.forward(&input).unwrap();
            let loss = criterion.loss(&pred, &target).unwrap();
            loss.backward();
            optimizer.step();
        }

        let pred = mlp.forward(&input).unwrap();
        assert!(approx::abs_diff_eq!(pred.to_vec()[0], 0., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[1], 1., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[2], 1., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[3], 1., epsilon=0.01));
    }

    #[test]
    fn test_train_xor_gate() {
        let input = Tensor::build([
            0., 0.,
            0., 1.,
            1., 0.,
            1., 1.,
        ], [4, 2]).unwrap().require_grad();

        let target = Tensor::build([
            0.,
            1.,
            1.,
            0.,
        ], [4, 1]).unwrap().require_grad();

        let mlp = MLP::from_archs([2, 4, 1].to_vec()).unwrap();
        let mut optimizer = Momentum::new(mlp.parameters(), 0.1, 0.1);
        let criterion = MSELoss::new();
    
        for _ in 0..5000 {
            optimizer.zero_grad();
            let pred = mlp.forward(&input).unwrap();
            let loss = criterion.loss(&pred, &target).unwrap();
            loss.backward();
            optimizer.step();
        }

        let pred = mlp.forward(&input).unwrap();
        assert!(approx::abs_diff_eq!(pred.to_vec()[0], 0., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[1], 1., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[2], 1., epsilon=0.01));
        assert!(approx::abs_diff_eq!(pred.to_vec()[3], 0., epsilon=0.01));
    }
}