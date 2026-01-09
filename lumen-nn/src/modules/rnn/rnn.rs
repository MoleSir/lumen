use lumen_core::{FloatDType, IndexOp, NumDType, Tensor, Var};
use lumen_macros::Module;
use crate::{init::Initialize, linear, Linear};

/// Creates a standard Elman RNN module with Tanh activation.
///
/// This factory function initializes the linear layers for input-to-hidden and 
/// hidden-to-hidden connections, along with a shared bias.
///
/// ## Arguments
///
/// * `input_size` - The number of expected features in the input `x`.
/// * `hidden_size` - The number of features in the hidden state `h`.
/// * `init` - The initialization scheme to use for the weights.
pub fn rnn<T: FloatDType>(input_size: usize, hidden_size: usize, init: &Initialize<T>) -> lumen_core::Result<Rnn<T>> {
    let input_proj = linear(input_size, hidden_size, false, init)?;
    let hidden_proj = linear(hidden_size, hidden_size, false, init)?;
    let bias = Var::zeros((1, 1, hidden_size,))?;
    Ok(Rnn::new(input_proj, hidden_proj, bias, input_size, hidden_size))
}

#[derive(Module)]
pub struct Rnn<T: NumDType> {
    pub input_proj: Linear<T>,
    pub hidden_proj: Linear<T>,
    pub bias: Tensor<T>,

    #[module(skip)]
    pub input_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<T: FloatDType> Rnn<T> {
    fn new(input_proj: Linear<T>, hidden_proj: Linear<T>, bias: Tensor<T>, input_size: usize, hidden_size: usize) -> Self {
        Self { input_proj, hidden_proj, bias, input_size, hidden_size }
    }

    /// Performs the forward pass for an entire sequence.
    ///
    /// This method processes a batch of sequences efficiently by pre-computing the input projections.
    ///
    /// ## Arguments
    ///
    /// * `input` - The input tensor of shape `(batch_size, seq_len, input_size)`.
    /// * `h0` - (Optional) The initial hidden state of shape `(batch_size, hidden_size)`.
    ///          If `None`, it defaults to a tensor of zeros.
    ///
    /// ## Returns
    ///
    /// A tuple `(output, h_n)` containing:
    /// * `output`: The hidden states for all time steps, shape `(batch_size, seq_len, hidden_size)`.
    /// * `h_n`: The final hidden state of the sequence, shape `(batch_size, hidden_size)`.
    pub fn forward(&self, input: &Tensor<T>, h0: Option<&Tensor<T>>) -> lumen_core::Result<(Tensor<T>, Tensor<T>)> {
        let (batch_size, seq_length, _input_size) = input.dims3()?;
        // let (batch_size, hidden_size) = h0.
        let h0 = match h0 {
            Some(h0) => h0.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        // (batch_size, seq_length, input_size) @ (input_size, hidden_size) => (batch_size, seq_length, hidden_size)
        let x_projed = self.input_proj.forward(&input)?;
        // (batch_size, seq_length, hidden_size) + 
        let x_projed = x_projed.broadcast_add(&self.bias)?;

        let mut outputs = vec![];
        let mut h_t = h0;
        for t in 0..seq_length {
            // x_t_projed: (batch_size, hidden_size)
            let x_t_projed = x_projed.index((.., t, ..))?;
            // h_t_projed: (batch_size, hidden_size)
            let h_t_projed = self.hidden_proj.forward(&h_t)?;

            // h_{t+1} = Tanh(x_t @ w_x + h_t @ w_h + bias)
            h_t = (x_t_projed + h_t_projed).tanh();
            
            outputs.push(h_t.clone());
        }     

        // [(batch_size, hidden_size); seq_len] => (batch_size, seq_len, hidden_size)
        let output = Tensor::stack(&outputs, 1)?;

        Ok((output, h_t))
    }

    /// Performs a single time-step forward pass (RNN Cell behavior).
    ///
    /// Unlike `forward`, this method processes 2D inputs for a single moment in time. 
    /// It is useful for auto-regressive tasks (e.g., language modeling generation, decoders).
    ///
    /// ## Arguments
    ///
    /// * `x` - The input tensor for the current time step, shape `(batch_size, input_size)`.
    /// * `h` - (Optional) The hidden state from the previous time step, shape `(batch_size, hidden_size)`.
    ///
    /// ## Returns
    ///
    /// * The new hidden state of shape `(batch_size, hidden_size)`.
    pub fn step(&self, x: &Tensor<T>, h: Option<&Tensor<T>>) -> lumen_core::Result<Tensor<T>> {
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
    use crate::{init::Initialize, Module};
    use super::rnn;

    #[test]
    fn test_init() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = rnn(4, 8, &init).unwrap();
        for (name, tensor) in rnn.named_params() {
            println!("{}: {}", name, tensor.shape());
        }
    }

    #[test]
    fn test_forward() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = rnn(4, 8, &init).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 10, 4)).unwrap();
        let (output, hidden_state) = rnn.forward(&input, None).unwrap();
        println!("{}", output.shape());
        println!("{}", hidden_state.shape());
    }
}