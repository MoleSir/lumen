use lumen_core::{FloatDType, Tensor};
use lumen_nn::{functional::LossReduction, optim::{Optimizer, SGD}, Linear, Module, MseLoss, Sigmoid};

#[derive(Module)]
pub struct Mlp<T: FloatDType> {
    pub linears: Vec<Linear<T>>,
    pub activates: Vec<Sigmoid>,
}

impl<T: FloatDType> Mlp<T> {
    pub fn from_archs(arch: impl AsRef<[usize]>) -> anyhow::Result<Self> {
        let arch = arch.as_ref();
        assert!(arch.len() >= 2);
        
        let mut linears = vec![];
        let mut activates = vec![];
        for (&in_dim, &out_dim) in arch.iter().zip(arch.iter().skip(1)) {
            linears.push(Linear::new(in_dim, out_dim, true, None)?);
            activates.push(Sigmoid::new());
        }
        
        Ok(Mlp { linears, activates })
    }

    pub fn layer_count(&self) -> usize {
        self.linears.len()
    }

    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let mut x = x.clone();
        for (linear, activate) in self.linears.iter().zip(self.activates.iter()) {
            x = linear.forward(&x)?;
            x = activate.forward(&x)?;
        }

        Ok(x)
    }
}

fn result_main() -> anyhow::Result<()> {
    let input = Tensor::new(&[
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])?;

    let target = Tensor::new(&[
        [0.], [0.], [0.], [1.]
    ])?;

    let mlp = Mlp::<f64>::from_archs([2, 4, 1])?;
    let mut optimizer = SGD::new(mlp.params(), 0.2);
    let criterion = MseLoss::new(Some(LossReduction::Mean));

    for _ in 0..5000 {
        let output = mlp.forward(&input)?;
        let loss = criterion.forward(&output, &target)?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }

    let output = mlp.forward(&input)?;
    println!("{}", output);

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprint!("Err: {}", e);
    }
}