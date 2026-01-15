use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_macros::Module;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnResult, Parameter};

#[derive(Module)]
pub struct Rnn<T: FloatDType> {
    pub input_proj: Linear<T>,
    pub hidden_proj: Linear<T>,
    pub bias: Parameter<T>,

    #[module(skip)]
    pub input_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

#[derive(Debug, derive_new::new)]
pub struct RnnConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<T: FloatDType> ModuleInit<T> for Rnn<T> {
    type Config = RnnConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let input_size = config.input_size;
        let hidden_size = config.hidden_size;

        let std = T::one() / T::from_usize(hidden_size).sqrt();
        let default_init = Init::uniform(-std, std);
        let init = init.unwrap_or(default_init);

        let input_proj = Linear::new(input_size, hidden_size, false, Some(init))?;
        let hidden_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let bias = init.init_param((1, hidden_size))?;

        Ok(Rnn { 
            input_proj, 
            hidden_proj, 
            bias, 
            input_size, 
            hidden_size 
        })
    }
}

impl<T: FloatDType> Rnn<T> {
    #[inline]
    pub fn new(input_size: usize, hidden_size: usize, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&RnnConfig::new(input_size, hidden_size), init)
    }

    pub fn forward(&self, input: &Tensor<T>, h0: Option<&Tensor<T>>) -> NnResult<(Tensor<T>, Tensor<T>)> {
        let (batch_size, seq_length, _input_size) = input.dims3()?;
        
        let h0 = match h0 {
            Some(h0) => h0.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        let x_projed = self.input_proj.forward(input)?;
        let x_projed = x_projed.broadcast_add(&self.bias)?;

        let mut outputs = vec![];
        let mut h_t = h0;
        
        for t in 0..seq_length {
            let x_t_projed = x_projed.index((.., t, ..))?;
            let h_t_projed = self.hidden_proj.forward(&h_t)?;

            h_t = (x_t_projed + h_t_projed).tanh();
            
            outputs.push(h_t.clone());
        }     

        let output = Tensor::stack(&outputs, 1)?;

        Ok((output, h_t))
    }

    pub fn step(&self, x: &Tensor<T>, h: Option<&Tensor<T>>) -> NnResult<Tensor<T>> {
        let (batch_size, _input_dim) = x.dims2()?;
        
        let h_prev = match h {
            Some(val) => val.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        let x_part = self.input_proj.forward(x)?;
        let h_part = self.hidden_proj.forward(&h_prev)?;
        
        let combined = (x_part + h_part).broadcast_add(&self.bias)?;
        
        Ok(combined.tanh())
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use crate::Module;
    use super::Rnn;

    #[test]
    fn test_init() {
        let rnn = Rnn::<f64>::new(4, 8, None).unwrap();
        for (name, tensor) in rnn.named_params() {
            println!("{}: {}", name, tensor.shape());
        }
    }

    #[test]
    fn test_forward() {
        let rnn = Rnn::<f64>::new(4, 8, None).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 10, 4)).unwrap();
        let (output, hidden_state) = rnn.forward(&input, None).unwrap();
        println!("{}", output.shape());
        println!("{}", hidden_state.shape());
    }
}