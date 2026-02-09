use anyhow::Context;
use lumen_core::{FloatDType, IndexOp, IntTensor, NoGradGuard, Tensor};
use lumen_dataset::{Batcher, DataLoader, Dataset};
use lumen_nn::{functional::LossReduction, optim::{Adam, AdamConfig, Optimizer}, CrossEntropyLoss, Embedding, Linear, Lstm, LstmState, Module};
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

    let dataset = Seq2SeqDataset::new(TOKEN_CONFIG, MAX_SEQ_LEN, MIN_SEQ_LEN, DATASET_LEN);
    let dataloader = Seq2SeqDataLoader::new(dataset, Seq2SeqBatcher::new(TOKEN_CONFIG), BATCH_SIZE, true);

    let mut model = Seq2Seq::<f32>::new(TOKEN_CONFIG, EMBED_SIZE, HIDDEN_SIZE).context("new model")?;
    let mut config = AdamConfig::default();
    config.lr = LR;
    let mut optimizer = Adam::new(model.params(), config)?;
    let criterion = CrossEntropyLoss::new(LossReduction::Mean);

    for epoch in 0..EPOCHS {
        for (batch_idx, batch) in dataloader.iter().enumerate() {
            // (batch_size, seq_len), (batch_size, seq_len+1), 
            let (src, trg) = batch.context("get batch")?;

            // (batch, seq_len+1, vocab_size)
            let output = model.forward(&src, Some(&trg)).context("model forward")?;

            // (batch, seq_len+1, vocab_size) => (batch * (seq_len+1), vocab_size)
            let output = output.flatten(0, 1)?;
            // (batch_size * (seq_len+1), 1)
            let trg = trg.flatten_all()?.unsqueeze(1)?;
            let loss = criterion.forward(&output, &trg).context("loss")?;
            
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

        let test_dataset = Seq2SeqDataset::new(TOKEN_CONFIG, MAX_SEQ_LEN, MIN_SEQ_LEN, 5);
        let test_loader = Seq2SeqDataLoader::new(test_dataset, Seq2SeqBatcher::new(TOKEN_CONFIG), 1, true);

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
    pub fn forward(&self, src: impl Into<IntTensor>) -> anyhow::Result<LstmState<T>> {
        // (batch_size, seq_len) => (batch_size, seq_len, embed_size)
        let embedded = self.embedding.forward(src).context("embedding forward")?;
        let (_, state) = self.rnn.forward(&embedded, None).context("rnn forward")?;
        Ok(state)
    } 
}

#[derive(Module)]
pub struct Decoder<T: FloatDType> {
    pub embedding: Embedding<T>,
    pub rnn: Lstm<T>,
    pub fc_out: Linear<T>,
}

impl<T: FloatDType> Decoder<T> {
    pub fn new(output_size: usize, embed_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        let embedding = Embedding::new(output_size, embed_size, None)?;
        let rnn = Lstm::new(embed_size, hidden_size, None)?;
        let fc_out = Linear::new(hidden_size, output_size, true, None)?;
        Ok(Self { embedding, rnn, fc_out })
    }

    /// Decoder forward
    /// 
    /// ## Args
    /// - input_token: (batch_size, 1)
    /// - state: from encoder / decoder
    pub fn forward(&self, input_token: impl Into<IntTensor>, state: &LstmState<T>) -> anyhow::Result<(Tensor<T>, LstmState<T>)> {
        // (batch_size, 1) => (batch_size, 1, embed_dim)
        let embedded = self.embedding.forward(input_token).context("embedding forward")?;
        // (batch_size, 1, hidden_size)
        let (output, state) = self.rnn.forward(&embedded, Some(state)).context("rnn forward")?;
        // (batch_size, 1, hidden_size) => (batch_size, 1, vocab_size)
        let prediction = self.fc_out.forward(&output).context("linear forward")?;

        Ok((prediction, state))
    }
}

#[derive(Module)]
pub struct Seq2Seq<T: FloatDType> {
    pub encoder: Encoder<T>,
    pub decoder: Decoder<T>,

    #[module(skip)]
    pub token_config: TokenConfig,
}

impl<T: FloatDType> Seq2Seq<T> {
    pub fn new(token_config: TokenConfig, embed_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        let encoder = Encoder::new(token_config.vocab_size, embed_size, hidden_size)?;
        let decoder = Decoder::new(token_config.vocab_size, embed_size, hidden_size)?;
        Ok(Self { token_config, encoder, decoder })
    }

    /// Forward pass
    /// 
    /// - src: (batch, seq_len)
    /// - trg: Option<&Tensor> -> (batch, seq_len + 1) [t1, t2, ..., EOS]
    pub fn forward(&self, src: &Tensor<u32>, trg: Option<&Tensor<u32>>) -> anyhow::Result<Tensor<T>> {
        /*

            src:
            +---+---+---+---+
            | 7 | 4 | 2 | P |
            +---+---+---+---+
            | 1 | 2 | 3 | 4 |
            +---+---+---+---+

            trg:
            +---+---+---+---+---+
            | 2 | 4 | 7 | E | P |
            +---+---+---+---+---+
            | 4 | 3 | 2 | 1 | E |
            +---+---+---+---+---+

        */
        let (batch, seq_len) = src.dims2()?;
        let mut outputs = vec![];

        // encoder
        let mut state = self.encoder.forward(src).context("encoder")?;
        let mut input_token = Tensor::new(self.token_config.sos_token)?.reshape((1, 1))?.repeat((batch, 1))?;
        
        for t in 0..seq_len + 1 {
            // (batch, 1, vocab_size)
            let (output, next_state) = self.decoder.forward(&input_token, &state).context("decoder")?;

            // update
            state = next_state;

            input_token = match trg {
                Some(trg) => trg.index((.., t))?.unsqueeze(1)?,
                None => output.argmax(2)?,    
            };

            // save
            outputs.push(output);
        }

        // [(batch_size, 1, vocab_size)] => (batch_size, seq_len+1, vocab_size)
        Tensor::cat(&outputs, 1).context("stack output")
    }

    /// src: (batch, seq_len)
    pub fn prediect(&self, src: &Tensor<u32>) -> anyhow::Result<Vec<Vec<u32>>> {
        // (batch_size, seq_len+1, vocab_size)
        let outputs = self.forward(&src, None).context("model forward")?;
        // (batch_size, seq_len+1)
        let pred_tokens = outputs.argmax(2).context("argmax output")?;

        let mut seqs = vec![];
        let batch = pred_tokens.dim(0)?;
        for b in 0..batch {
            let tokens = pred_tokens.index(b)?.to_vec()?;
            seqs.push(tokens);
        }

        Ok(seqs)
    }
}

pub struct Seq2SeqDataset {
    pub token_config: TokenConfig,
    pub max_seq_len: usize,
    pub min_seq_len: usize,
    pub dataset_len: usize,
}

impl Seq2SeqDataset {
    pub fn new(token_config: TokenConfig, max_seq_len: usize, min_seq_len: usize, dataset_len: usize) -> Self {
        assert!(max_seq_len >= min_seq_len, "max_seq_len < min_seq_len");
        Self { token_config, max_seq_len, min_seq_len,  dataset_len }
    }
}

impl Dataset for Seq2SeqDataset {
    type Item = (Vec<u32>, Vec<u32>);
    
    fn get(&self, _index: usize) -> Option<Self::Item> {
        let mut rng = rng();
        
        // rand a seq_len
        let seq_len = rng.random_range(self.min_seq_len..=self.max_seq_len);

        // generate input tokens
        let src: Vec<u32> = (0..seq_len).map(|_| rng.random_range(self.token_config.min_token..=self.token_config.max_token)).collect();
        
        // revser input and add eos
        let mut trg = src.clone();
        trg.reverse();
        trg.push(self.token_config.eos_token);

        Some((src, trg))
    }

    fn len(&self) -> usize {
        self.dataset_len
    }
}

pub struct Seq2SeqBatcher {
    pub token_config: TokenConfig,
}

impl Seq2SeqBatcher {
    pub fn new(token_config: TokenConfig) -> Self {
        Self { token_config }
    }
}

impl Batcher for Seq2SeqBatcher {
    type Item = (Vec<u32>, Vec<u32>);
    type Output = Result<(Tensor<u32>, Tensor<u32>), lumen_core::Error>;

    fn batch(&self, items: Vec<(Vec<u32>, Vec<u32>)>) -> Self::Output {
        let mut xs = vec![];
        let mut ys = vec![];
        
        // find max seq len!
        let mut max_seq_len = 0;
        for (x, y) in items.iter() {
            assert_eq!(x.len() + 1, y.len());
            if x.len() > max_seq_len {
                max_seq_len = x.len();
            }
        }

        for (mut x, mut y) in items {
            // add pad!
            for _ in x.len()..max_seq_len {
                x.push(self.token_config.pad_token);
            }

            for _ in y.len()..max_seq_len+1 {
                y.push(self.token_config.pad_token);
            }

            assert_eq!(x.len(), max_seq_len);
            assert_eq!(y.len(), max_seq_len + 1);

            xs.push(Tensor::new(x)?);
            ys.push(Tensor::new(y)?);
        }

        let xs = Tensor::stack(&xs, 0)?;
        let ys = Tensor::stack(&ys, 0)?; 
        Ok((xs, ys))
    }
}

pub type Seq2SeqDataLoader = DataLoader<Seq2SeqDataset, Seq2SeqBatcher>;
