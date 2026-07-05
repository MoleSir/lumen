use std::convert::Infallible;

use anyhow::Context;
use lumen_core::{FloatDType, IndexOp, IntTensor, NoGradGuard, Tensor};
use lumen_dataset::{Batcher, DataLoader, Dataset};
use lumen_nn::{functional::LossReduction, optim::{Adam, AdamConfig, Optimizer}, CrossEntropyLoss, Embedding, Linear, Lstm, LstmState, Module, Parameter};
use rand::{rng, Rng};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}

#[derive(Debug, Clone)]
pub struct TokenConfig {
    pub pad_token: u32,
    pub min_token: u32,
    pub max_token: u32,
    pub sos_token: u32,
    pub eos_token: u32,
    pub vocab_size: usize,
}

impl TokenConfig {
    pub const fn new(min_token: u32, max_token: u32) -> Self {
        assert!(max_token > min_token, "max token <= min token");
        Self {
            min_token, max_token,
            pad_token: max_token + 1,
            sos_token: max_token + 2,
            eos_token: max_token + 3,
            vocab_size: (max_token + 3 - min_token + 1) as usize,
        }
    }
}
 
fn result_main() -> anyhow::Result<()> {
    const TOKEN_CONFIG: TokenConfig = TokenConfig::new(0, 10);

    const EMBED_SIZE: usize = 16;
    const HIDDEN_SIZE: usize = 32;
    const MAX_SEQ_LEN: usize = 8;
    const MIN_SEQ_LEN: usize = 4;
    const LR: f32 = 0.01;
    const EPOCHS: usize = 20;
    const DATASET_LEN: usize = 1000;
    const BATCH_SIZE: usize = 10;

    let dataset = PointerNetDataset::new(TOKEN_CONFIG, MAX_SEQ_LEN, MIN_SEQ_LEN, DATASET_LEN);
    let dataloader = PointerNetDataLoader::new(dataset, PointerNetBatcher::new(TOKEN_CONFIG), BATCH_SIZE, true);

    let mut model = PointerNet::<f32>::new(TOKEN_CONFIG, EMBED_SIZE, HIDDEN_SIZE).context("new model")?;
    let mut config = AdamConfig::default();
    config.lr = LR;
    let mut optimizer = Adam::new(model.params(), config)?;
    let criterion = CrossEntropyLoss::new(LossReduction::Mean);

    for epoch in 0..EPOCHS {
        for (batch_idx, batch) in dataloader.iter().enumerate() {
            // (batch_size, seq_len), (batch_size, seq_len), 
            let (src, trg_indices) = batch.context("get batch")?;

            // (batch, seq_len, seq_len)
            let logitss = model.forward(&src, Some(&trg_indices)).context("model forward")?;

            // (batch, seq_len, seq_len) => (batch * seq_len, seq_len)
            let logitss = logitss.flatten(0, 1)?;
            // (batch_size * seq_len, 1)
            let trg_indices = trg_indices.flatten_all()?.unsqueeze(1)?;
            let loss = criterion.forward(&logitss, &trg_indices).context("loss")?;
            
            let grads = loss.backward().context("backward")?;
            optimizer.step(&grads).context("step")?;

            if batch_idx % 10 == 0 {
                println!("epoch {}, batch {}, loss = {}", epoch, batch_idx, loss.to_scalar()?);
            }
        }
    }

    {
        let _guard = NoGradGuard::new();
        model.eval();

        let test_dataset = PointerNetDataset::new(TOKEN_CONFIG, MAX_SEQ_LEN, MIN_SEQ_LEN, 5);
        let test_loader = PointerNetDataLoader::new(test_dataset, PointerNetBatcher::new(TOKEN_CONFIG), 1, true);

        for (i, batch) in test_loader.iter().enumerate() {
            // (1, seq_len), (1, seq_len+1)
            let (src, trg) = batch.context("batch")?;
            // (1, seq_len) => (1, seq_len+1)
            let seqs = model.prediect(&src).context("prediect")?;

            println!("Batch {}:", i);
            println!("Pred: {:?}", seqs[0]);
            println!("Anwe: {:?}", &trg.flatten_all()?.to_vec()?);
        }
    }

    Ok(())
}

#[derive(Module)]
pub struct Encoder<T: FloatDType> {
    pub embedding: Embedding<T>,
    pub rnn: Lstm<T>,
}

impl<T: FloatDType> Encoder<T> {
    pub fn new(input_size: usize, embed_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        let embedding = Embedding::new(input_size, embed_size, None)?;
        let rnn = Lstm::new(embed_size, hidden_size, None)?;
        Ok(Self { embedding, rnn })
    }

    /// src: (batch_size, seq_len)
    pub fn forward(&self, src: impl Into<IntTensor>) -> anyhow::Result<(Tensor<T>, LstmState<T>)> {
        // (batch_size, seq_len) => (batch_size, seq_len, embed_size)
        let embedded = self.embedding.forward(src).context("embedding forward")?;
        let (output, state) = self.rnn.forward(&embedded, None).context("rnn forward")?;
        Ok((output, state))
    } 
}

#[derive(Module)]
pub struct Decoder<T: FloatDType> {
    pub rnn: Lstm<T>,
    pub attention: PointerAttention<T>,
}

impl<T: FloatDType> Decoder<T> {
    pub fn new(hidden_size: usize) -> anyhow::Result<Self> {
        let rnn = Lstm::new(hidden_size, hidden_size, None)?;
        let attention = PointerAttention::new(hidden_size)?;
        
        Ok(Self { rnn, attention })
    }

    /// Decoder forward
    /// 
    /// ## Args
    /// - input: (batch_size, 1, hidden_size)
    /// - encoder_outputs: (batch, seq_len, hidden_size)
    /// - state: from encoder / decoder
    pub fn forward(
        &self, 
        input: &Tensor<T>, 
        encoder_outputs: &Tensor<T>,
        state: &LstmState<T>,
        mask: &Tensor<bool>,
    ) -> anyhow::Result<(Tensor<T>, LstmState<T>)> {
        // (batch_size, 1, hidden_size)
        let (output, state) = self.rnn.forward(&input, Some(state)).context("rnn forward")?;
        // (batch, seq_len)
        let logits = self.attention.forward(&output, &encoder_outputs, mask).context("attention forward")?;

        Ok((logits, state))
    }
}

#[derive(Module)]
pub struct PointerAttention<T: FloatDType> {
    pub w_enc: Linear<T>,
    pub w_dec: Linear<T>,
    pub v: Linear<T>,
}

impl<T: FloatDType> PointerAttention<T> {
    pub fn new(hiddent_size: usize) -> anyhow::Result<Self> {
        Ok(Self {
            w_enc: Linear::new(hiddent_size, hiddent_size, false, None)?,
            w_dec: Linear::new(hiddent_size, hiddent_size, false, None)?,
            v: Linear::new(hiddent_size, 1, false, None)?,
        })
    }

    pub fn forward(
        &self, 
        decoder_state: &Tensor<T>, 
        encoder_outputs: &Tensor<T>,
        mask: &Tensor<bool>,
    ) -> anyhow::Result<Tensor<T>> {
        // 1. Projection
        // (batch, seq_len, hidden_size) => (batch, seq_len, hidden_size)
        let enc_proj = self.w_enc.forward(encoder_outputs).context("w_enc")?;
        // (batch, 1, hidden_size) => (batch, 1, hidden_size)
        let dec_proj = self.w_dec.forward(&decoder_state).context("w_dec")?;

        // 2. Activate
        // (batch, seq_len, hidden_size)
        let energy = enc_proj.broadcast_add(&dec_proj)?.tanh()?;

        // 3. Score
        // (batch, seq_len, 1) => (batch, seq_len)
        let scores = self.v.forward(&energy)?.squeeze(2)?;
        let masked_scores = mask.if_else(scores, T::MIN_VALUE)?;

        Ok(masked_scores)
    }
}

#[derive(Module)]
pub struct PointerNet<T: FloatDType> {
    pub encoder: Encoder<T>,
    pub decoder: Decoder<T>,
    pub sos_param: Parameter<T>,

    #[module(skip)]
    pub token_config: TokenConfig,
}

impl<T: FloatDType> PointerNet<T> {
    pub fn new(token_config: TokenConfig, embed_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        let encoder = Encoder::new(token_config.vocab_size, embed_size, hidden_size)?;
        let decoder = Decoder::new(hidden_size)?;
        let sos_param = Parameter::new(Tensor::randn(T::zero(), T::one(), (1, hidden_size))?);
        Ok(Self { token_config, encoder, sos_param, decoder })
    }

    /// Forward pass
    /// 
    /// - src: (batch, seq_len)
    /// - trg_indices: Option<&Tensor> -> (batch, seq_len)
    pub fn forward(&self, src: &Tensor<u32>, trg_indices: Option<&Tensor<u32>>) -> anyhow::Result<Tensor<T>> {
        let (batch, seq_len) = src.dims2()?;
        let mut logitss = vec![];
        let mask = src.ne(self.token_config.pad_token)?;

        // encoder
        let (encoder_outputs, mut state) = self.encoder.forward(src).context("encoder")?;
        // (1, hidden_size) => (batch, 1, hidden_size)
        let mut input = self.sos_param.unsqueeze(0)?.repeat_dim(0, batch)?;
        
        for t in 0..seq_len {
            // (batch, seq_len)
            let (logits, next_state) = self.decoder.forward(&input, &encoder_outputs, &state, &mask).context("decoder")?;

            // update
            state = next_state;

            // (batch)
            let input_index = match trg_indices {
                Some(trg_indices) => trg_indices.index((.., t))?,
                None => logits.argmax(1)?,
            };

            let (_, _, hidden_size) = encoder_outputs.dims3()?;
            let idx_expanded = input_index
                .reshape((batch, 1, 1))? // (batch, 1, 1)
                .repeat_dim(2, hidden_size)?;   // (batch, 1, hidden)

            // (batch, 1, hidden_size)
            input = encoder_outputs.gather(&idx_expanded, 1).context("gather next input")?;

            // save
            logitss.push(logits);
        }

        // [(batch_size, seq_len)] => (batch_size, seq_len, seq_len)
        Tensor::stack(&logitss, 1).context("stack output")
    }

    /// src: (batch, seq_len)
    pub fn prediect(&self, src: &Tensor<u32>) -> anyhow::Result<Vec<Vec<u32>>> {
        // (batch_size, seq_len, seq_len)
        let logitss = self.forward(&src, None).context("model forward")?;
        // (batch_size, seq_len)
        let pred_tokens = logitss.argmax(2).context("argmax output")?;

        let mut seqs = vec![];
        let batch = pred_tokens.dim(0)?;
        for b in 0..batch {
            let tokens = pred_tokens.index(b)?.to_vec()?;
            seqs.push(tokens);
        }

        Ok(seqs)
    }
}

pub struct PointerNetDataset {
    pub token_config: TokenConfig,
    pub max_seq_len: usize,
    pub min_seq_len: usize,
    pub dataset_len: usize,
}

impl PointerNetDataset {
    pub fn new(token_config: TokenConfig, max_seq_len: usize, min_seq_len: usize, dataset_len: usize) -> Self {
        assert!(max_seq_len >= min_seq_len, "max_seq_len < min_seq_len");
        Self { token_config, max_seq_len, min_seq_len,  dataset_len }
    }
}

impl Dataset for PointerNetDataset {
    type Item = (Vec<u32>, Vec<u32>);
    type Error = Infallible;
    
    fn get(&self, _index: usize) -> Result<Option<Self::Item>, Self::Error> {
        let mut rng = rng();
        
        // rand a seq_len
        let seq_len = rng.random_range(self.min_seq_len..=self.max_seq_len);

        // generate input tokens
        let mut src: Vec<u32> = (0..seq_len)
            .map(|_| rng.random_range(self.token_config.min_token..=self.token_config.max_token))
            .collect();
        
        let mut trg_indices: Vec<_> = src.iter()
            .cloned()
            .enumerate()
            .collect();
        trg_indices.sort_by_key(|(_, v)| *v);
        let mut trg_indices: Vec<_> = trg_indices.into_iter()
            .map(|(i, _)| i as u32)
            .collect();

        src.push(self.token_config.eos_token);
        trg_indices.push(seq_len as u32);

        Ok(Some((src, trg_indices)))
    }

    fn len(&self) -> usize {
        self.dataset_len
    }
}

pub struct PointerNetBatcher {
    pub token_config: TokenConfig,
}

impl PointerNetBatcher {
    pub fn new(token_config: TokenConfig) -> Self {
        Self { token_config }
    }
}

impl Batcher for PointerNetBatcher {
    type Item = (Vec<u32>, Vec<u32>);
    type Output = (Tensor<u32>, Tensor<u32>);
    type Error = lumen_core::Error;

    fn batch(&self, items: Vec<(Vec<u32>, Vec<u32>)>) -> lumen_core::Result<Self::Output> {
        let mut xs = vec![];
        let mut ys = vec![];
        
        // find max seq len!
        let mut max_seq_len = 0;
        for (x, y) in items.iter() {
            assert_eq!(x.len(), y.len());
            if x.len() > max_seq_len {
                max_seq_len = x.len();
            }
        }

        for (mut x, mut y) in items {
            // add pad!
            for _ in x.len()..max_seq_len {
                x.push(self.token_config.min_token);
            }

            for _ in y.len()..max_seq_len {
                y.push(self.token_config.min_token);
            }

            assert_eq!(x.len(), max_seq_len);
            assert_eq!(y.len(), max_seq_len);

            xs.push(Tensor::new(x)?);
            ys.push(Tensor::new(y)?);
        }

        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?; 
        Ok((xs, ys))
    }
}

pub type PointerNetDataLoader = DataLoader<PointerNetDataset, PointerNetBatcher>;
