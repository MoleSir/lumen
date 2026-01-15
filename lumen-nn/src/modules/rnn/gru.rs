use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_macros::Module;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnResult};

#[derive(Module)]
pub struct Gru<T: FloatDType> {
    pub input_proj: Linear<T>,
    pub hidden_proj: Linear<T>,

    #[module(skip)]
    pub input_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

#[derive(Debug, derive_new::new)]
pub struct GruConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl<T: FloatDType> ModuleInit<T> for Gru<T> {
    type Config = GruConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let input_size = config.input_size;
        let hidden_size = config.hidden_size;
        let gate_size = 3 * hidden_size;

        // PyTorch default init: uniform(-std, std) where std = 1 / sqrt(hidden_size)
        let std = T::one() / T::from_usize(hidden_size).sqrt();
        let default_init = Init::uniform(-std, std);
        let init = init.unwrap_or(default_init);

        // Linear layers with bias enabled
        let input_proj = Linear::new(input_size, gate_size, true, Some(init))?;
        let hidden_proj = Linear::new(hidden_size, gate_size, true, Some(init))?;

        Ok(Gru { 
            input_proj, 
            hidden_proj, 
            input_size, 
            hidden_size 
        })
    }
}

impl<T: FloatDType> Gru<T> {
    /// Creates a GRU module.
    ///
    /// This factory function inits the linear layers for input-to-hidden and 
    /// hidden-to-hidden connections.
    ///
    /// ## Arguments
    ///
    /// * `input_size` - The number of expected features in the input `x`.
    /// * `hidden_size` - The number of features in the hidden state `h`.
    /// * `init` - The initialization scheme to use for the weights.
    #[inline]
    pub fn new(input_size: usize, hidden_size: usize) -> NnResult<Self> {
        Self::init(&GruConfig::new(input_size, hidden_size), None)
    }

    #[inline]
    pub fn new_with(input_size: usize, hidden_size: usize, init: Init<T>) -> NnResult<Self> {
        Self::init(&GruConfig::new(input_size, hidden_size), Some(init))
    }

    /// Performs the forward pass for an entire sequence.
    ///
    /// This method optimizes the computation by pre-calculating the input projections 
    /// for all time steps before iterating through the sequence.
    ///
    /// ## Arguments
    ///
    /// * `input` - The input tensor of shape `(batch_size, seq_len, input_size)`.
    /// * `h0` - (Optional) The initial hidden state of shape `(batch_size, hidden_size)`.
    ///          If `None`, defaults to a tensor of zeros.
    ///
    /// ## Returns
    ///
    /// A tuple `(output, h_n)` containing:
    /// * `output`: The hidden states for all time steps, shape `(batch_size, seq_len, hidden_size)`.
    /// * `h_n`: The final hidden state of the sequence, shape `(batch_size, hidden_size)`.
    pub fn forward(&self, input: &Tensor<T>, h0: Option<&Tensor<T>>) -> NnResult<(Tensor<T>, Tensor<T>)> {
        let (batch_size, seq_length, _input_size) = input.dims3()?;
        
        let h0 = match h0 {
            Some(h0) => h0.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        // Pre-compute input projections: (batch_size, seq_length, 3 * hidden_size)
        let x_projed = self.input_proj.forward(input)?;

        let mut outputs = vec![];
        let mut h_t = h0;

        for t in 0..seq_length {
            // (batch_size, 3 * hidden_size)
            let x_t_all = x_projed.index((.., t, ..))?;
            // (batch_size, 3 * hidden_size)
            let h_t_all = self.hidden_proj.forward(&h_t)?;
            
            h_t = self.gru_cell(&x_t_all, &h_t_all, &h_t)?;

            outputs.push(h_t.clone());
        }     

        let output = Tensor::stack(&outputs, 1)?;
        Ok((output, h_t))
    }

    /// Performs a single time-step forward pass (GRU Cell).
    ///
    /// This is useful for processing data step-by-step, such as in auto-regressive 
    /// generation tasks or when the sequence length is not known in advance.
    ///
    /// ## Arguments
    ///
    /// * `x` - The input tensor for the current time step, shape `(batch_size, input_size)`.
    /// * `h` - (Optional) The hidden state from the previous time step, shape `(batch_size, hidden_size)`.
    ///
    /// ## Returns
    ///
    /// * The new hidden state of shape `(batch_size, hidden_size)`.
    pub fn step(&self, x: &Tensor<T>, h: Option<&Tensor<T>>) -> NnResult<Tensor<T>> {
        let (batch_size, _input_dim) = x.dims2()?;
        let h_prev = match h {
            Some(val) => val.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        // (batch_size, input_dim) => (batch_size, 3 * hidden_size)
        let x_all = self.input_proj.forward(x)?;
        // (batch_size, hidden_size) => (batch_size, 3 * hidden_size)
        let h_all = self.hidden_proj.forward(&h_prev)?;

        let h_next = self.gru_cell(&x_all, &h_all, &h_prev)?;

        Ok(h_next)
    }

    fn gru_cell(&self, x_t_all: &Tensor<T>, h_t_all: &Tensor<T>, h_prev: &Tensor<T>) -> NnResult<Tensor<T>> {
        let h_size = self.hidden_size;

        // Split into Reset (r), Update (z), New (n)
        // (batch_size, 3 * hidden_size) => (batch_size, hidden_size)
        let x_t_r = x_t_all.narrow(1, 0 * h_size, h_size)?;
        let x_t_z = x_t_all.narrow(1, 1 * h_size, h_size)?;
        let x_t_n = x_t_all.narrow(1, 2 * h_size, h_size)?;

        let h_t_r = h_t_all.narrow(1, 0 * h_size, h_size)?;
        let h_t_z = h_t_all.narrow(1, 1 * h_size, h_size)?;
        let h_t_n = h_t_all.narrow(1, 2 * h_size, h_size)?;

        // r_t = sigmoid(W_ir*x + b_ir + W_hr*h + b_hr)
        let r_t = (x_t_r + h_t_r).sigmoid();
        
        // z_t = sigmoid(W_iz*x + b_iz + W_hz*h + b_hz)
        let z_t = (x_t_z + h_t_z).sigmoid();
        
        // n_t = tanh(W_in*x + b_in + r_t * (W_hn*h + b_hn))
        let n_t = (x_t_n + (r_t * h_t_n)).tanh();

        // h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        let h_next = ((T::one() - &z_t) * n_t) + (z_t * h_prev);

        Ok(h_next)
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use crate::Module;
    use super::Gru;

    #[test]
    fn test_init() {
        let gru = Gru::<f64>::new(4, 8).unwrap();
        for (name, tensor) in gru.named_params() {
            println!("{}: {}", name, tensor.shape());
        }
    }

    #[test]
    fn test_forward() {
        let gru = Gru::<f64>::new(4, 8).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 10, 4)).unwrap();
        let (output, hidden_state) = gru.forward(&input, None).unwrap();
        println!("{}", output.shape());
        println!("{}", hidden_state.shape());
    }

    #[test]
    fn test_step() {
        let gru = Gru::<f64>::new(4, 8).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 4)).unwrap();
        let hidden_state = gru.step(&input, None).unwrap();
        println!("{}", hidden_state.shape());
    }
}