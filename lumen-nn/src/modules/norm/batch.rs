use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;

use crate::{init::Init, Buffer, ModuleInit, NnCtxError, NnError, NnResult, Parameter};

/// Applies Batch Normalization over a 2D or 3D input.
#[derive(Module)]
#[module(display = "format")]
#[module(train = "train")]
pub struct BatchNorm1d<T: FloatDType> {
    pub gamma: Parameter<T>,
    pub beta: Parameter<T>,

    pub running_mean: Buffer<T>,
    pub running_var: Buffer<T>,

    #[module(skip)]
    pub num_features: usize,
    #[module(skip)]
    pub eps: T,
    #[module(skip)]
    pub momentum: T,
    #[module(skip)]
    pub training: bool,
}

#[derive(derive_new::new)]
pub struct BatchNorm1dConfig {
    pub num_features: usize,
    pub eps: f64,
    pub momentum: f64,
}

impl<T: FloatDType> ModuleInit<T> for BatchNorm1d<T> {
    type Config = BatchNorm1dConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let gamma = init.unwrap_or(Init::ones()).init_param((1, config.num_features))?;
        let beta = init.unwrap_or(Init::zeros()).init_param((1, config.num_features))?;

        let running_mean = init.unwrap_or(Init::zeros()).init_buffer((1, config.num_features))?;
        let running_var = init.unwrap_or(Init::ones()).init_buffer((1, config.num_features))?;

        Ok(Self {
            gamma, beta,
            running_mean, running_var,

            num_features: config.num_features,
            eps: T::from_f64(config.eps),
            momentum: T::from_f64(config.momentum),
            training: true,
        })
    }
}

impl<T: FloatDType> BatchNorm1d<T> {
    pub fn new(num_features: usize, eps: f64, momentum: f64, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&BatchNorm1dConfig::new(num_features, eps, momentum), init)
    }

    /// BatchNorm1d forward
    /// 
    /// ## Argument
    /// 
    /// * `input`: (N, C, L) or (N, L, C)
    pub fn forward(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> { 
        match input.rank() {
            2 => self.forward_impl(input),
            3 => {
                // (N, C, L) => (N, L, C)
                let input_permuted = input.permute((0, 2, 1))?;
                let (n, l, c) = input_permuted.dims3()?;
                let input_flattened = input_permuted.reshape((n * l, c))?;

                let out_flattened = self.forward_impl(&input_flattened)?;

                let out = out_flattened
                    .reshape((n, l, c))?
                    .permute((0, 2, 1))?;

                Ok(out)
            }
            _ => Err(NnError::BatchNorm1dUnsupportShape(input.shape().clone()))?,
        }
    }

    fn forward_impl(&self, x: &Tensor<T>) -> NnResult<Tensor<T>> {
        assert_eq!(x.rank(), 2);

        // x shape: (N, D)
        let x_normalized = if self.training {
            let batch_mean = x.mean_keepdim(0)?; // (1, D)
            let batch_var = x.var_keepdim(0)?;   // (1, D)

            // x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            let x_normalized = x
                .broadcast_sub(&batch_mean)?
                .broadcast_div(&(&batch_var + self.eps).sqrt())?; // (N, D)
            
            // self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_mean.mul_(self.momentum)?;
            self.running_mean.add_((T::one() - self.momentum) * batch_mean)?;

            // self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            self.running_var.mul_(self.momentum)?;
            self.running_var.add_((T::one() - self.momentum) * batch_var)?;

            x_normalized
        } else {
            // x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            let x_normalized = x
                .broadcast_sub(&self.running_mean)?
                .broadcast_div(&self.running_var.add(self.eps)?.sqrt())?; // (N, D)

            x_normalized
        };

        // out = self.gamma * x_normalized + self.beta
        let out = self.gamma.broadcast_mul(&x_normalized)?.broadcast_add(&self.beta)?;
        Ok(out)
    }

    fn format(&self) -> String {
        format!("num_features={}, eps={}, momentum={}", self.num_features, self.eps, self.momentum)
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use crate::Module;

    use super::BatchNorm1d;

    #[test]
    fn test_forward() {
        let mut ln = BatchNorm1d::<f64>::new(10, 1e-5, 0.8, None).unwrap();
        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());
        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());
        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());
        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());

        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10, 5)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());

        ln.train(false);

        let input = Tensor::<f64>::randn(0.0, 1.0, (5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap().shape());
    }

    #[test]
    fn test_state_param_buffer() {
        let ln = BatchNorm1d::<f64>::new(10, 1e-5, 0.8, None).unwrap();

        println!("Parameter:");
        for (name, param) in ln.named_params() {
            println!("{}: {}", name, param.shape());
        }
        println!("");
        assert_eq!(ln.param_count(), 2);

        println!("Buffer:");
        for (name, buffer) in ln.named_buffers() {
            println!("{}: {}", name, buffer.shape());
        }
        println!("");
        assert_eq!(ln.buffer_count(), 2);

        println!("State:");
        for (name, state) in ln.named_dyn_states() {
            println!("{}: {}", name, state.shape());
        }
        println!("");
        assert_eq!(ln.state_count(), 4);
    }
}