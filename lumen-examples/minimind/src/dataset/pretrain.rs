use std::{cell::RefCell, fs::File, io::{BufRead, BufReader, Seek, SeekFrom}, path::Path};
use anyhow::Context;
use lumen_core::{FloatDType, Tensor};
use lumen_dataset::Dataset;
use lumen_nn::functional::LossReduction;
use tokenizers::Tokenizer;

pub const IGNORE_ID: u32 = u32::MAX;

pub struct PretrainDataset {
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    bos_id: u32, 
    eos_id: u32,
    pad_token_id: u32,
    jsonl_dataset: RefCell<JsonlDataset>,
}

impl PretrainDataset {
    pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Tokenizer, max_length: usize) -> anyhow::Result<Self> {
        let jsonl_dataset = JsonlDataset::new(data_path).context("new jsonl dataset")?;
        let bos_id = tokenizer.token_to_id("<|im_start|>").expect("BOS token not found");
        let eos_id = tokenizer.token_to_id("<|im_end|>").expect("EOS token not found");
        let pad_token_id = tokenizer.token_to_id("<|endoftext|>").expect("EOS token not found");

        Ok(Self {
            tokenizer,
            max_length,
            bos_id,
            eos_id,
            pad_token_id,
            jsonl_dataset: RefCell::new(jsonl_dataset),
        })
    }
}

impl Dataset for PretrainDataset {
    type Item = (Tensor<u32>, Tensor<u32>);

    fn get(&self, index: usize) -> Option<Self::Item> {
        let value = self.jsonl_dataset.borrow_mut().get(index)?;
        let text = value
            .as_object().expect("need a json object")
            .get("text").expect("need text feiled")
            .as_str().expect("text should be str");

        let encoding = self.tokenizer.encode(text, false).ok()?;
        let mut tokens = encoding.get_ids().to_vec();
        let limit = self.max_length.saturating_sub(2);
        if tokens.len() > limit {
            tokens.truncate(limit);
        }
        let mut final_tokens = Vec::with_capacity(tokens.len() + 2);
        final_tokens.push(self.bos_id);
        final_tokens.extend(tokens);
        final_tokens.push(self.eos_id);

        let mut input_ids = final_tokens;
        let mut labels = input_ids.clone();

        if input_ids.len() < self.max_length {
            for _ in 0..self.max_length - input_ids.len() {
                input_ids.push(self.pad_token_id);
                labels.push(IGNORE_ID);
            }
        }

        let input_ids = Tensor::new(input_ids).ok()?;
        let labels = Tensor::new(labels).ok()?;

        Some((input_ids, labels))
    }

    fn len(&self) -> usize {
        self.jsonl_dataset.borrow().len()
    }
}

pub fn cross_entropy_with_ignore<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<u32>,
    reduction: LossReduction,
) -> anyhow::Result<Tensor<T>> {
    let (safe_target, mask) = {
        // 为 IGNORE_ID -> false -> 填充 0
        let valid_mask = target.ne(IGNORE_ID)?;
        let safe_target = valid_mask.if_else(target, 0)?;
        let float_mask = valid_mask.cast::<T>()?; 
        (safe_target, float_mask)
    };

    let log_probs = lumen_nn::functional::log_softmax(input, 1)?;
    let gathered = log_probs.gather(safe_target, 1)?;
    let mut loss = gathered.neg()?;

    loss = loss.mul(&mask)?;

    match reduction {
        LossReduction::None => Ok(loss),
        LossReduction::Sum => Ok(loss.sum_all()?),
        LossReduction::Mean => {
            let sum_loss = loss.sum_all()?;
            let valid_count = mask.sum_all()?;
            Ok(sum_loss.div(&valid_count)?)
        }
    }
}

pub struct JsonlDataset {
    file: File,
    offsets: Vec<u64>,
}

impl JsonlDataset {
    pub fn new<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&file);

        let mut offsets = vec![];
        let mut current_offset = 0;
        let mut line_buffer = String::new();

        loop {
            offsets.push(current_offset);
            let bytes_read = reader.read_line(&mut line_buffer)?;
            if bytes_read == 0 {
                offsets.pop();
                break;
            }
            current_offset += bytes_read as u64;
            line_buffer.clear();
        }

        file.seek(SeekFrom::Start(0))?;
        Ok(Self { file, offsets })
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn get(&mut self, index: usize) -> Option<serde_json::Value> {
        if index >= self.offsets.len() {
            return None;
        }

        self.file.seek(SeekFrom::Start(self.offsets[index])).ok()?;

        let mut reader = BufReader::new(&self.file);
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        
        serde_json::from_str(&line).ok()
    }
}