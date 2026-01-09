use lumen_core::{FloatDType, IndexOp, NumDType, Tensor};
use lumen_macros::Module;
use crate::{init::Initialize, linear, Linear};

/// Creates a GRU module.
///
/// This factory function initializes the linear layers for input-to-hidden and 
/// hidden-to-hidden connections.
///
/// ## Arguments
///
/// * `input_size` - The number of expected features in the input `x`.
/// * `hidden_size` - The number of features in the hidden state `h`.
/// * `init` - The initialization scheme to use for the weights.
pub fn gru<T: FloatDType>(input_size: usize, hidden_size: usize, init: &Initialize<T>) -> lumen_core::Result<Gru<T>> {
    let gate_size = 3 * hidden_size;
    
    let input_proj = linear(input_size, gate_size, true, init)?;
    let hidden_proj = linear(hidden_size, gate_size, true, init)?;
    
    Ok(Gru::new(input_proj, hidden_proj, input_size, hidden_size))
}

#[derive(Module)]
pub struct Gru<T: NumDType> {
    pub input_proj: Linear<T>,
    pub hidden_proj: Linear<T>,

    #[module(skip)]
    pub input_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<T: FloatDType> Gru<T> {
    fn new(
        input_proj: Linear<T>, 
        hidden_proj: Linear<T>, 
        input_size: usize, 
        hidden_size: usize
    ) -> Self {
        Self { input_proj, hidden_proj, input_size, hidden_size }
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
    pub fn forward(&self, input: &Tensor<T>, h0: Option<&Tensor<T>>) -> lumen_core::Result<(Tensor<T>, Tensor<T>)> {
        let (batch_size, seq_length, _input_size) = input.dims3()?;
        
        let h0 = match h0 {
            Some(h0) => h0.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        // (batch_size, seq_length, input_size) => (batch_size, seq_length, 3 * hidden_size)
        let x_projed = self.input_proj.forward(&input)?;

        let mut outputs = vec![];
        let mut h_t = h0;

        for t in 0..seq_length {
            // (batch_size, 3 * hidden_size)
            let x_t_all = x_projed.index((.., t, ..))?;
            // (batch_size, hidden_size) => (batch_size, 3 * hidden_size)
            let h_t_all = self.hidden_proj.forward(&h_t)?;
            
            h_t = self.gru_step(&x_t_all, &h_t_all, &h_t)?;

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
    pub fn step(&self, x: &Tensor<T>, h: Option<&Tensor<T>>) -> lumen_core::Result<Tensor<T>> {
        let (batch_size, _input_dim) = x.dims2()?;
        let h_prev = match h {
            Some(val) => val.clone(),
            None => Tensor::zeros((batch_size, self.hidden_size))?,
        };

        // (batch_size, input_dim) => (batch_size, 3 * hidden_size)
        let x_all = self.input_proj.forward(x)?;
        // (batch_size, hidden_size) => (batch_size, 3 * hidden_size)
        let h_all = self.hidden_proj.forward(&h_prev)?;

        let h_next = self.gru_step(&x_all, &h_all, &h_prev)?;

        Ok(h_next)
    }

    /// Internal helper to calculate the GRU gating logic.
    ///
    /// It splits the projected tensors into Reset (r), Update (z), and New (n) components
    /// and applies the GRU equations.
    ///
    /// ## Equations
    ///
    /// * `r_t = sigmoid(x_r + h_r)`
    /// * `z_t = sigmoid(x_z + h_z)`
    /// * `n_t = tanh(x_n + (r_t * h_n))`
    /// * `h_t = (1 - z_t) * n_t + z_t * h_{t-1}`
    fn gru_step(&self, x_t_all: &Tensor<T>, h_t_all: &Tensor<T>, h_prev: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // x_t_all: (batch_size, 3 * hidden_size)
        // h_t_all: (batch_size, 3 * hidden_size)
        // h_prev: (batch_size, hidden_size)
        let h_size = self.hidden_size;

        // (batch_size, hidden_size)
        let x_t_r = x_t_all.narrow(1, 0 * h_size, h_size)?;
        let x_t_z = x_t_all.narrow(1, 1 * h_size, h_size)?;
        let x_t_n = x_t_all.narrow(1, 2 * h_size, h_size)?;

        // (batch_size, hidden_size)
        let h_t_r = h_t_all.narrow(1, 0 * h_size, h_size)?;
        let h_t_z = h_t_all.narrow(1, 1 * h_size, h_size)?;
        let h_t_n = h_t_all.narrow(1, 2 * h_size, h_size)?;

        // Gates
        let r_t = (x_t_r + h_t_r).sigmoid(); // (batch_size, hidden_size)
        let z_t = (x_t_z + h_t_z).sigmoid(); // (batch_size, hidden_size)
        
        // Candidate
        let n_t = (x_t_n + (r_t * h_t_n)).tanh(); // (batch_size, hidden_size)

        // Output
        let h_next = ((T::one() - &z_t) * n_t) + (z_t * h_prev); // (batch_size, hidden_size)

        Ok(h_next)
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use crate::{init::Initialize, Module};
    use super::gru;

    #[test]
    fn test_init() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = gru(4, 8, &init).unwrap();
        for (name, tensor) in rnn.named_params() {
            println!("{}: {}", name, tensor.shape());
        }
    }

    #[test]
    fn test_forward() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = gru(4, 8, &init).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 10, 4)).unwrap();
        let (output, hidden_state) = rnn.forward(&input, None).unwrap();
        println!("{}", output.shape());
        println!("{}", hidden_state.shape());
    }

    #[test]
    fn test_step() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = gru(4, 8, &init).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 4)).unwrap();
        let hidden_state = rnn.step(&input, None).unwrap();
        println!("{}", hidden_state.shape());
    }
}