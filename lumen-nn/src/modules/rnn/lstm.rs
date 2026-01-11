use lumen_core::{FloatDType, IndexOp, NumDType, Tensor};
use lumen_macros::Module;
use crate::{init::Initialize, linear, Linear};

/// Creates an LSTM module.
///
/// This factory function initializes the linear layers.
///
/// ## Arguments
///
/// * `input_size` - The number of features in the input `x`.
/// * `hidden_size` - The number of features in the hidden state `h` and cell state `c`.
/// * `init` - The initialization scheme.
pub fn lstm<T: FloatDType>(input_size: usize, hidden_size: usize, init: &Initialize<T>) -> lumen_core::Result<Lstm<T>> {
    // LSTM needs 4 parts: Input(i), Forget(f), Gate/Cell(g), Output(o).
    // Total dimension: 4 * hidden_size
    let gate_size = 4 * hidden_size;
    
    // linear(..., true, ...) enables internal bias (b_ih, b_hh)
    let input_proj = linear(input_size, gate_size, true, init)?;
    let hidden_proj = linear(hidden_size, gate_size, true, init)?;
    
    Ok(Lstm::new(input_proj, hidden_proj, input_size, hidden_size))
}

#[derive(Module)]
pub struct Lstm<T: FloatDType> {
    pub input_proj: Linear<T>,
    pub hidden_proj: Linear<T>,

    #[module(skip)]
    pub input_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

/// Helper struct to manage the LSTM state tuple.
/// This isn't strictly necessary but makes signatures cleaner.
#[derive(Clone)]
pub struct LstmState<T: NumDType> {
    pub h: Tensor<T>,
    pub c: Tensor<T>,
}

impl<T: FloatDType> Lstm<T> {
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
    /// ## Arguments
    ///
    /// * `input` - Input tensor `(batch_size, seq_len, input_size)`.
    /// * `state` - Optional initial state `(h_0, c_0)`.
    ///
    /// ## Returns
    ///
    /// * `output` - Sequence of hidden states `(batch_size, seq_len, hidden_size)`.
    /// * `final_state` - The final `(h_n, c_n)`.
    pub fn forward(&self, input: &Tensor<T>, state: Option<&LstmState<T>>) -> lumen_core::Result<(Tensor<T>, LstmState<T>)> {
        let (batch_size, seq_length, _input_size) = input.dims3()?;
        
        // Initialize states
        let (mut h_t, mut c_t) = match state {
            Some(s) => (s.h.clone(), s.c.clone()),
            None => {
                let h = Tensor::zeros((batch_size, self.hidden_size))?;
                let c = Tensor::zeros((batch_size, self.hidden_size))?;
                (h, c)
            }
        };

        // Optimization: Pre-compute input projections for all time steps
        // (batch_size, seq_length, 4 * hidden_size)
        let x_projed = self.input_proj.forward(&input)?;

        let mut outputs = vec![];

        for t in 0..seq_length {
            // Extract projected input for time step t: (batch_size, 4 * hidden_size)
            let x_t_all = x_projed.index((.., t, ..))?;
            
            // Compute hidden projections: (batch_size, 4 * hidden_size)
            let h_t_all = self.hidden_proj.forward(&h_t)?;
            
            let (h_next, c_next) = self.lstm_cell(&x_t_all, &h_t_all, &c_t)?;
            
            h_t = h_next;
            c_t = c_next;

            outputs.push(h_t.clone());
        }     

        // Stack hidden states: (batch_size, seq_len, hidden_size)
        let output = Tensor::stack(&outputs, 1)?;
        
        Ok((output, LstmState { h: h_t, c: c_t }))
    }

    /// Performs a single time-step forward pass.
    ///
    /// ## Arguments
    ///
    /// * `x` - Input at current step `(batch_size, input_size)`.
    /// * `state` - Previous state `(h_{t-1}, c_{t-1})`.
    pub fn step(&self, x: &Tensor<T>, state: Option<&LstmState<T>>) -> lumen_core::Result<LstmState<T>> {
        let (batch_size, _input_dim) = x.dims2()?;
        
        let (h_prev, c_prev) = match state {
            Some(s) => (s.h.clone(), s.c.clone()),
            None => {
                let h = Tensor::zeros((batch_size, self.hidden_size))?;
                let c = Tensor::zeros((batch_size, self.hidden_size))?;
                (h, c)
            }
        };

        // (batch_size, 4 * hidden_size)
        let x_all = self.input_proj.forward(x)?;
        let h_all = self.hidden_proj.forward(&h_prev)?;

        let (h_next, c_next) = self.lstm_cell(&x_all, &h_all, &c_prev)?;

        Ok(LstmState { h: h_next, c: c_next })
    }

    /// Internal calculation logic for the LSTM cell.
    ///
    /// We assume the weights are organized in **IFGO** order (Input, Forget, Gate, Output),
    /// which is a common convention (e.g., in PyTorch).
    fn lstm_cell(&self, x_all: &Tensor<T>, h_all: &Tensor<T>, c_prev: &Tensor<T>) -> lumen_core::Result<(Tensor<T>, Tensor<T>)> {
        let h_size = self.hidden_size;

        // Split Input projections
        let x_i = x_all.narrow(1, 0 * h_size, h_size)?; // Input Gate
        let x_f = x_all.narrow(1, 1 * h_size, h_size)?; // Forget Gate
        let x_g = x_all.narrow(1, 2 * h_size, h_size)?; // Cell Candidate (Gate)
        let x_o = x_all.narrow(1, 3 * h_size, h_size)?; // Output Gate

        // Split Hidden projections
        let h_i = h_all.narrow(1, 0 * h_size, h_size)?;
        let h_f = h_all.narrow(1, 1 * h_size, h_size)?;
        let h_g = h_all.narrow(1, 2 * h_size, h_size)?;
        let h_o = h_all.narrow(1, 3 * h_size, h_size)?;

        // Compute Gates
        let i_t = (x_i + h_i).sigmoid(); // Input gate
        let f_t = (x_f + h_f).sigmoid(); // Forget gate
        let g_t = (x_g + h_g).tanh();    // Cell candidate
        let o_t = (x_o + h_o).sigmoid(); // Output gate

        // Update Cell State
        // c_t = f_t * c_{t-1} + i_t * g_t
        let c_next = (f_t * c_prev) + (i_t * g_t);

        // Update Hidden State
        // h_t = o_t * tanh(c_t)
        let h_next = o_t * c_next.tanh();

        Ok((h_next, c_next))
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use crate::{init::Initialize, Module};
    use super::lstm;

    #[test]
    fn test_init() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = lstm(4, 8, &init).unwrap();
        for (name, tensor) in rnn.named_params() {
            println!("{}: {}", name, tensor.shape());
        }
    }

    #[test]
    fn test_forward() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = lstm(4, 8, &init).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 10, 4)).unwrap();
        let (output, state) = rnn.forward(&input, None).unwrap();
        println!("{}", output.shape());
        println!("{}", state.c.shape());
        println!("{}", state.h.shape());
    }

    #[test]
    fn test_step() {
        let init = Initialize::<f64>::standard_uniform();
        let rnn = lstm(4, 8, &init).unwrap();
        let input = Tensor::randn(0.0, 1.0, (1, 4)).unwrap();
        let state = rnn.step(&input, None).unwrap();
        println!("{}", state.c.shape());
        println!("{}", state.h.shape());
    }
}