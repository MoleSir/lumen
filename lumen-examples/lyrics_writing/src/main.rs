use std::{collections::HashMap, path::Path};
use anyhow::Context;
use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_dataset::{DataLoader, Dataset, TensorPairBatcher};
use lumen_nn::{init::Initialize, optim::{AdamW, AdamWConfig, Optimizer}, Embedding, Linear, Lstm, Module, Sigmoid};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    const EMBED_SIZE: usize = 32;
    const HIDDEN_SIZE: usize = 128;
    const LEARN_RATE: f64 = 0.001;
    const BATCH_SIZE: usize = 32;
    const EPOCHS: usize = 15;
    const SEQ_LEN: usize = 48;
    
    let dataset = LyricsDataset::init(SEQ_LEN, "./lyrics.txt")?;
    let vocab_size = dataset.word2index.len();
    println!("get dataset with vocab size = {}", vocab_size);

    let (train_dataset, test_dataset) = lumen_dataset::transform::random_split(dataset, 0.1);
    let train_loader = DataLoader::new(train_dataset, TensorPairBatcher::default(), BATCH_SIZE, true);
    let _test_loader = DataLoader::new(test_dataset, TensorPairBatcher::default(), BATCH_SIZE, true);

    let model = LyricsNet::<f64>::init(vocab_size, EMBED_SIZE, HIDDEN_SIZE)?;
    let mut optimizer = AdamW::new(model.params(), AdamWConfig::default())?;
    optimizer.config.lr = LEARN_RATE;
    let cirterion = lumen_nn::CrossEntropyLoss;

    println!("begin training!");
    for epoch in 0..EPOCHS {
        for (i, batch) in train_loader.iter().enumerate() {
            let (input, target) = batch.context("batch dataset")?; // (batch_size, seq_len)
            let output = model.forward(&input).context("model forward")?; // (batch_size, seq_len, vocab_size)
            let (batch_size, seq_len) = input.dims2()?;
            let loss = cirterion.forward(
                &output.reshape((batch_size * seq_len, vocab_size))?, 
                &target.reshape((batch_size * seq_len, 1))?
            ).context("get loss")?;

            let grads = loss.backward().context("backward")?;
            optimizer.step(&grads).context("update params")?;

            let acc = accuracy(&output, &target).context("cal accuracy")?;

            println!(
                "Training: Epoch={}, Batch={}/{}, Loss={}, Accuracy={}",
                epoch, i, train_loader.batch_count(), loss.sum_all()?.to_scalar()?, acc
            )  
        }
    }

    Ok(())
}

// ========================================================================================================== //
//                                            LyricsNet     
// ========================================================================================================== //

#[derive(Module)]
pub struct LyricsNet<T: FloatDType> {
    pub embedding: Embedding<T>,
    pub lstm: Lstm<T>,
    pub h2h: Linear<T>,
    pub act: Sigmoid,
    pub h2o: Linear<T>,
}

impl<T: FloatDType> LyricsNet<T> {
    pub fn init(vocab_size: usize, embed_size: usize, hidden_size: usize) -> anyhow::Result<Self> {
        let initialize = Initialize::<T>::uniform(T::zero(), T::from_f64(0.1));
        let embedding = lumen_nn::embedding(vocab_size, embed_size, &initialize)?;
        let lstm = lumen_nn::lstm(embed_size, hidden_size, &initialize)?;
        let h2h = lumen_nn::linear(hidden_size, hidden_size, true, &initialize)?;
        let h2o = lumen_nn::linear(hidden_size, vocab_size, true, &initialize)?;

        Ok(Self { embedding, lstm, h2h, act: Sigmoid::new(), h2o })
    }

    pub fn forward(&self, word_ids: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
        // word_ids: (batch_size, seq_len)
        // (batch_size, seq_len) => (batch_size, seq_len, embed_size)
        let embedded = self.embedding.forward(word_ids).context("embedding forward")?;
        // (batch_size, seq_len, embed_size) => (batch_size, seq_len, hidden_size)
        let (lstm_out, _) = self.lstm.forward(&embedded, None).context("lstm_out forward")?;
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        let out = self.h2h.forward(&lstm_out).context("h2h forward")?;
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, vocab_size)
        let out = self.act.forward(&out).context("act forward")?;
        let out = self.h2o.forward(&out).context("h2o forward")?;

        Ok(out)
    }
}

pub fn accuracy<T: FloatDType>(output: &Tensor<T>, target: &Tensor<u32>) -> anyhow::Result<T> {
    let _guard = lumen_core::NoGradGuard::new();
    // output: (batch_size, seq_len, vocab_size) 
    // target: (batch_size, seq_len, vocab_size) 
    // (batch_size, seq_len, vocab_size) => (batch_size, seq_len) 
    let pred = output.argmax(2)?;
    let correct = pred.eq(target)?;
    let acc = correct.to_dtype::<T>().mean_all()?.to_scalar()?;
    Ok(acc)
}

// ========================================================================================================== //
//                                            LyricsDataset     
// ========================================================================================================== //

pub struct LyricsDataset {
    pub seq_len: usize,
    pub word2index: HashMap<char, u32>,
    pub index2word: HashMap<u32, char>,
    pub data: Tensor<u32>,
}

impl LyricsDataset {
    pub fn init<P: AsRef<Path>>(seq_len: usize, path: P) -> anyhow::Result<Self> {
        const SOS: u32 = 0;
        const EOS: u32 = 1;

        let contexts = std::fs::read_to_string(path)?;

        let mut word2index = HashMap::new();
        word2index.insert('\0', SOS); 
        word2index.insert('\x01', EOS);

        let mut indices = vec![];
        let mut num_words = 0;
        for line in contexts.lines() {
            indices.push(SOS);
            for word in line.chars() {
                let idx = if let Some(id) = word2index.get(&word) {
                    *id
                } else {
                    let new_id = num_words;
                    word2index.insert(word, new_id);
                    num_words += 1;
                    new_id
                };
                indices.push(idx);
            }
            indices.push(EOS);
        }

        let index2word = word2index.iter().map(|(word, idx)| (*idx, *word)).collect();
        let data = Tensor::new(indices)?;

        Ok(Self { seq_len, word2index, index2word, data })
    }
}

impl Dataset for LyricsDataset {
    type Item = (Tensor<u32>, Tensor<u32>);

    fn len(&self) -> usize {
        if self.data.element_count() <= 1 { return 0; }
        (self.data.element_count() - 1) / self.seq_len
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }

        let start = index * self.seq_len;
        let end = start + self.seq_len;

        let input = self.data.index(start..end).unwrap(); // (seq_len, )
        let output = self.data.index(start+1..end+1).unwrap(); // (seq_len, )

        Some((input, output))
    }
}


