use std::path::Path;

use anyhow::Context;
use lumen_core::Tensor;
use lumen_dataset::{common::JsonlDataset, Dataset};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::model::IGNORE_ID;

pub struct PretrainDataset {
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    bos_id: u32, 
    eos_id: u32,
    pad_token_id: u32,
    jsonl_dataset: JsonlDataset<PretrainItem>,
}

impl PretrainDataset {
    pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Tokenizer, max_length: usize) -> anyhow::Result<Self> {
        let jsonl_dataset = JsonlDataset::new(data_path).context("new jsonl dataset")?;
        let bos_id = tokenizer.token_to_id("<|im_start|>").expect("BOS token not found");
        let eos_id = tokenizer.token_to_id("<|im_end|>").expect("EOS token not found");
        let pad_token_id = tokenizer.token_to_id("<|endoftext|>").expect("PAD token not found");

        Ok(Self {
            tokenizer,
            max_length,
            bos_id,
            eos_id,
            pad_token_id,
            jsonl_dataset,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct PretrainItem {
    text: String,
}

impl Dataset for PretrainDataset {
    type Item = (Tensor<u32>, Tensor<u32>);
    type Error = anyhow::Error;

    fn get(&self, index: usize) -> anyhow::Result<Option<Self::Item>> {
        let value = self.jsonl_dataset.get(index)?;
        let value = match value {
            Some(v) => v,
            None => return Ok(None),
        };

        let encoding = self.tokenizer
            .encode(value.text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let mut tokens = encoding.get_ids().to_vec();
        let limit = self.max_length.saturating_sub(2);
        if tokens.len() > limit {
            tokens.truncate(limit);
        }
        let mut final_tokens = Vec::with_capacity(tokens.len() + 2);
        final_tokens.push(self.bos_id);
        final_tokens.extend(tokens);
        final_tokens.push(self.eos_id);

        let mut input_ids = final_tokens[0..final_tokens.len() - 1].to_vec();
        let mut labels = final_tokens[1..final_tokens.len()].to_vec();

        let target_len = self.max_length - 1; 
        if input_ids.len() < target_len {
            for _ in 0..(target_len - input_ids.len()) {
                input_ids.push(self.pad_token_id);
                labels.push(IGNORE_ID);
            }
        }

        let input_ids = Tensor::new(input_ids)?;
        let labels = Tensor::new(labels)?;

        Ok(Some((input_ids, labels)))
    }

    fn len(&self) -> usize {
        self.jsonl_dataset.len()
    }
}

#[cfg(test)]
mod test {
    use lumen_dataset::{common::JsonlDataset, Dataset};
    use tokenizers::Tokenizer;
    use super::{PretrainDataset, PretrainItem};

    #[test]
    fn test_jsonl_dataset() {
        let dataset = JsonlDataset::<PretrainItem>::new("./assets/cache/pretrain_hq.jsonl").unwrap();
        println!("{}", dataset.len());
        println!("{}", dataset.get(10000).unwrap().unwrap().text);
    }

    #[test]
    fn test_pretrain_dataset() {
        let tokenizer = Tokenizer::from_file("./assets/tokenizer.json").unwrap();
        let dataset = PretrainDataset::new(
            "./assets/cache/pretrain_hq.jsonl", tokenizer, 512,
        ).unwrap();

        println!("{}", dataset.len());
        let (input_ids, label) = dataset.get(512).unwrap().unwrap();
        println!("{}", input_ids);
        println!("{}", label);
    }
}