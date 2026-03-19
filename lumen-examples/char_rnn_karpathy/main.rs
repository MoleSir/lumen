use std::collections::{HashMap, HashSet};
use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_dataset::Dataset;
use lumen_nn::functional::LossReduction;
use lumen_nn::optim::{Optimizer, SGD};
use lumen_nn::{init::Init,  Module};
use lumen_nn::{CrossEntropyLoss, Parameter};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}
 
fn result_main() -> anyhow::Result<()> {
    const HIDDEN_SIZE: usize = 64;
    const EPOCH: usize = 20;
    const SEQ_LENGTH: usize = 25;
    const STRIDE: usize = SEQ_LENGTH;
    
    // Data
    let data = r"
hello world
hello deep learning
deep neural networks
neural networks learn patterns
patterns in data
data drives learning
learning from examples
examples help networks
networks process information
information is everywhere
everywhere you look data
";
    let dataset = TextDataset::new(data.to_string(), SEQ_LENGTH, STRIDE);
    let rnn = VanillaRNN::<f32>::new(dataset.vocab_size, HIDDEN_SIZE)?;
    let mut optimizer = SGD::new(rnn.params(), 0.1);
    let lossor = CrossEntropyLoss::new(LossReduction::Mean);

    let mut hprev = Tensor::zeros((HIDDEN_SIZE, 1))?;

    for _ in 0..EPOCH {
        for (b, batch) in dataset.iter().enumerate() {
            let (inputs, targets) = batch?;
            let state = rnn.forward(&inputs, &hprev)?;
            hprev = state.hs.get(&(inputs.len() as i32 - 1)).expect("get hpre").clone();

            let outputs = Tensor::stack(&state.ys.values().cloned().collect::<Vec<_>>(), 0)?.squeeze(2)?;
            let targets = Tensor::from_vec(targets.into_iter().map(|v| v as i32).collect::<Vec<i32>>(), (inputs.len(), 1))?;
            
            let loss = lossor.forward(&outputs, targets)?;
            
            if b % 10 == 0 {
                println!("{}", loss.to_scalar()?);
            }
            let grads = loss.backward()?;
        
            optimizer.step(&grads)?;
        }
    }

    Ok(())
}

pub struct TextDataset {
    pub vocab_size: usize,
    pub seq_length: usize,
    pub stride: usize,
    pub data: String,
    pub chars: Vec<char>,
    pub char_to_ix: HashMap<char, usize>,
    pub ix_to_char: HashMap<usize, char>,
}

impl TextDataset {
    pub fn new(data: String, seq_length: usize, stride: usize) -> Self {
        let mut chars = data.chars().collect::<HashSet<char>>().into_iter().collect::<Vec<char>>();
        chars.sort();
        let vocab_size = chars.len();
        
        let ix_to_char = chars.iter().cloned().enumerate().collect::<HashMap<usize, char>>();
        let char_to_ix = chars.iter().cloned().enumerate().map(|(k, v)| (v, k)).collect::<HashMap<char, usize>>();
    
        Self { vocab_size, seq_length, stride, data, chars, char_to_ix, ix_to_char }
    }

    pub fn encode(&self, input: &str) -> anyhow::Result<Vec<usize>> {
        input.chars()
            .map(|c| {
                self.char_to_ix
                    .get(&c)
                    .ok_or_else(|| anyhow::anyhow!("un support char '{}'", c))
                    .cloned()
            })
            .collect()
    }

    pub fn decode(&self, input: &[usize]) -> anyhow::Result<String> {
        input.iter()
            .map(|c| {
                self.ix_to_char
                    .get(&c)
                    .ok_or_else(|| anyhow::anyhow!("un support index '{}'", c))
                    .cloned()
        })
        .collect()
    }
}

impl Dataset for TextDataset {
    type Item = (Vec<usize>, Vec<usize>);
    type Error = anyhow::Error;

    fn get(&self, index: usize) -> anyhow::Result<Option<Self::Item>> {
        if index >= self.len() {
            Ok(None)
        } else {
            let start = index * self.stride;
            let end = start + self.seq_length;
            
            let input = &self.data[start..end];
            let target = &self.data[start+1..end+1];
            
            let input = self.encode(input)?;
            let target = self.encode(target)?;

            Ok(Some((input, target)))
        }
    }

    fn len(&self) -> usize {
        (self.data.len() - self.seq_length + 1) / self.stride 
    }
}

#[derive(Module)]
#[module(display = "display")]
pub struct VanillaRNN<T: FloatDType> {
    pub wxh: Parameter<T>,
    pub whh: Parameter<T>,
    pub why: Parameter<T>,
    pub bh: Parameter<T>,
    pub by: Parameter<T>,

    #[module(skip)]
    pub vocab_size: usize,
    #[module(skip)]
    pub hidden_size: usize,
}

pub struct State<T: FloatDType> {
    pub xs: HashMap<i32, Tensor<T>>,
    pub hs: HashMap<i32, Tensor<T>>,
    pub ys: HashMap<i32, Tensor<T>>,
    pub ps: HashMap<i32, Tensor<T>>,
}

impl<T: FloatDType> VanillaRNN<T> {
    pub fn new(vocab_size: usize, hidden_size: usize,) -> anyhow::Result<Self> {
        let init = Init::standard_normal();
        Ok(Self {
            wxh: init.init_param((hidden_size, vocab_size))?,
            whh: init.init_param((hidden_size, hidden_size))?,
            why: init.init_param((vocab_size, hidden_size))?,
            bh: Init::Zeros.init_param((hidden_size, 1))?,
            by: Init::Zeros.init_param((vocab_size, 1))?,
            vocab_size,
            hidden_size
        })
    }

    pub fn forward(&self, inputs: &[usize], h_prev: &Tensor<T>) -> anyhow::Result<State<T>> {
        let mut xs: HashMap<i32, Tensor<T>> = HashMap::new();
        let mut hs = HashMap::new();
        let mut ys = HashMap::new();
        let mut ps = HashMap::new();

        hs.insert(-1, h_prev.copy()?);

        for (t, &char_idx) in inputs.iter().enumerate() {
            let x = Tensor::zeros((self.vocab_size, 1))?;
            x.index((char_idx, 0))?.set_scalar(T::one())?;
            xs.insert(t as i32, x.clone());
            
            let h = hs.get(&(t as i32 -1)).expect("get last");
            let h_next = self.calculate_hidden(&x, h)?;
            hs.insert(t as i32, h_next.clone());
            
            let y = self.calculate_output(&h_next)?;
            ys.insert(t as i32, y.clone());

            let p = lumen_nn::functional::softmax(&y, 0)?;
            ps.insert(t as i32, p);
        }

        Ok(State { xs, hs, ys, ps })
    }

    pub fn loss(self, ps: &HashMap<i32, Tensor<T>>, targets: &[usize]) -> anyhow::Result<T> {
        let mut loss = T::zero();
        for (t, &target_idx) in targets.iter().enumerate() {
            let p = ps.get(&(t as i32)).expect("no target");
            loss += p.index((target_idx, 0))?.to_scalar()?;
        }
        Ok(loss)
    }

    pub fn sample(&self, h: &Tensor<T>, seed_ix: usize, n: usize) -> anyhow::Result<Vec<usize>> {
        let mut indices = vec![];

        let mut x = Tensor::zeros((self.vocab_size, 1))?;
        x.index((seed_ix, 0))?.set_scalar(T::one())?;
        let mut h = h.clone();

        for _ in 0..n {
            h = self.calculate_hidden(&x, &h)?;
            let y = self.calculate_output(&h)?;
            let p = lumen_nn::functional::softmax(&y, 0)?;

            let ix = p.unsqueeze(1)?.argmax(0)?.to_scalar()? as usize;

            x = Tensor::zeros((self.vocab_size, 1))?;
            x.index((seed_ix, 0))?.set_scalar(T::one())?;

            indices.push(ix);
        }        

        Ok(indices)
    }

    fn calculate_hidden(&self, x: &Tensor<T>, h: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        (
            self.wxh.matmul(&x)? +
            self.whh.matmul(&h)? + 
            self.bh.tensor()
        ).tanh()
    }

    fn calculate_output(&self, h: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        Ok( self.why.matmul(h)? + self.by.tensor() )
    }

    fn display(&self) -> String {
        format!("vocab_size={}, hidden_size={}", self.vocab_size, self.hidden_size)
    }
}