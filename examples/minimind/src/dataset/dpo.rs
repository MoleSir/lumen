use std::{path::Path, sync::Arc};
use anyhow::Context;
use lumen_core::Tensor;
use lumen_dataset::{common::JsonlDataset, Batcher, DataLoader, Dataset};
use serde::Deserialize;
use crate::tokenizer::{EncodeOptions, Tokenizer};
use super::types::Message;

pub type DopDataLoader = DataLoader<DpoDataset, DpoBatcher>;

pub struct DpoDataset {
    pub tokenizer: Arc<Tokenizer>,
    pub max_length: usize,
    pub assistant_bos_ids: Vec<u32>,
    pub assistant_eos_ids: Vec<u32>,
    jsonl_dataset: JsonlDataset<DpoDatasetItem>,
}

#[derive(Debug, Deserialize)]
pub struct DpoDatasetItem {
    chosen: Vec<Message>,
    rejected: Vec<Message>,
}

impl DpoDataset {
    pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Arc<Tokenizer>, max_length: usize) -> anyhow::Result<Self> {
        let jsonl_dataset = JsonlDataset::new(data_path).context("new jsonl dataset")?;

        // assistant 回复的开始/结束序列：<bos>assistant\n"xxxxx"<eos>\n
        let assistant_bos_ids = tokenizer
            .encode(&format!("{}assistant\n", tokenizer.bos_token()), EncodeOptions::default())?
            .get_ids()
            .to_vec();
        let assistant_eos_ids = tokenizer
            .encode(&format!("{}\n", tokenizer.eos_token()), EncodeOptions::default())?
            .get_ids()
            .to_vec();

        Ok(Self {
            tokenizer,
            max_length,
            assistant_bos_ids,
            assistant_eos_ids,
            jsonl_dataset,
        })
    }

    /// 找出 input_ids 中的 <bos>assistant\n[.....]\n<eos> 部分，设置 mask 为 1，其他的为 0
    fn generate_loss_mask(&self, input_ids: &[u32]) -> anyhow::Result<Vec<bool>> {
        let mut loss_mask = vec![false; input_ids.len()];
        let mut i = 0usize;
        while i+self.assistant_bos_ids.len() < input_ids.len() {
            if input_ids[i..i+self.assistant_bos_ids.len()] == self.assistant_bos_ids {
                let start = i + self.assistant_bos_ids.len();
                let mut end = start;
                while end < input_ids.len() {
                    if input_ids[end..end+self.assistant_eos_ids.len()] == self.assistant_eos_ids {
                        break;
                    }
                    end += 1;
                }
                
                for j in start..(end+self.assistant_eos_ids.len()).min(self.max_length) {
                    loss_mask[j] = true;
                }

                i = if end < input_ids.len() {
                    end + self.assistant_eos_ids.len()
                } else {
                    input_ids.len()
                };
            } else {
                i += 1;
            }
        }
        Ok(loss_mask[1..].to_vec())
    }

    fn shift_tokens(mut input_ids: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
        let y: Vec<_> = input_ids[1..].to_vec();
        input_ids.pop().unwrap();
        return (input_ids, y)
    }
}

impl Dataset for DpoDataset {
    type Error = anyhow::Error;
    type Item = DpoItem;

    fn len(&self) -> usize {
        self.jsonl_dataset.len()
    }

    fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error> {
        let v = self.jsonl_dataset.get(index)?;
        let v = match v {
            Some(v) => v,
            None => return Ok(None),
        };

        let chosen = v.chosen;
        let rejected = v.rejected;

        // 使用 messages 数组创建对话模板
        let chosen_prompt = self.tokenizer.apply_chat_template(&chosen, false);
        let rejected_prompt = self.tokenizer.apply_chat_template(&rejected, false);

        // 使用 encode 转为 tokens
        let mut options = EncodeOptions::default();
        options.max_length = Some(self.max_length);
        options.padding = true;
        let chosen_tokens = self.tokenizer.encode(&chosen_prompt, options.clone())?.get_ids().to_vec(); // (seq)
        let rejecten_tokens = self.tokenizer.encode(&rejected_prompt, options.clone())?.get_ids().to_vec(); // (seq)

        // 此时的 tokens 是全部输入的 token，需要增加一个 mask，使得只训练 assistant 的输出，不训练 user prompt
        let chosen_mask = self.generate_loss_mask(&chosen_tokens)?; // (seq-1)
        let rejecten_mask = self.generate_loss_mask(&rejecten_tokens)?; // (seq-1)

        // 对 chosen 进行偏移
        let (chosen_x, chosen_y) = Self::shift_tokens(chosen_tokens); // (seq-1)
        let (rejecten_x, rejecten_y) = Self::shift_tokens(rejecten_tokens); // (seq-1)

        Ok(Some(DpoItem {
            chosen_x,
            chosen_y,
            chosen_mask,

            rejecten_x,
            rejecten_y,
            rejecten_mask,
        }))
    }
}

#[derive(Default)]
pub struct DpoBatcher {

}

impl Batcher for DpoBatcher {
    type Error = anyhow::Error;
    type Item = DpoItem;
    type Output = DpoTensorItem;

    fn batch(&self, items: Vec<Self::Item>) -> Result<Self::Output, Self::Error> {
        let batch_size = items.len();
        
        let mut cxs = vec![];
        let mut cys = vec![];
        let mut cms = vec![];

        let mut rxs = vec![];
        let mut rys = vec![];
        let mut rms = vec![];

        for item in items {
            // cxs.push(Tensor::new(item.chosen_x)?);
            // cys.push(Tensor::new(item.chosen_y)?);
            // cms.push(Tensor::new(item.chosen_mask)?);
            // rxs.push(Tensor::new(item.rejecten_x)?);
            // rys.push(Tensor::new(item.rejecten_y)?);
            // rms.push(Tensor::new(item.rejecten_mask)?);
            cxs.extend(item.chosen_x);
            cys.extend(item.chosen_y);
            cms.extend(item.chosen_mask);
            rxs.extend(item.rejecten_x);
            rys.extend(item.rejecten_y);
            rms.extend(item.rejecten_mask);
        }

        // 都是 batch, seq 的 shape
        Ok(DpoTensorItem {
            // chosen_x: Tensor::stack(&cxs, 0)?,
            // chosen_y: Tensor::stack(&cys, 0)?,
            // chosen_mask: Tensor::stack(&cms, 0)?,

            // rejecten_x: Tensor::stack(&rxs, 0)?,
            // rejecten_y: Tensor::stack(&rys, 0)?,
            // rejecten_mask: Tensor::stack(&rms, 0)?,
            chosen_x: Tensor::new(cxs)?.reshape((batch_size, ()))?,
            chosen_y: Tensor::new(cys)?.reshape((batch_size, ()))?,
            chosen_mask: Tensor::new(cms)?.reshape((batch_size, ()))?,

            rejecten_x: Tensor::new(rxs)?.reshape((batch_size, ()))?,
            rejecten_y: Tensor::new(rys)?.reshape((batch_size, ()))?,
            rejecten_mask: Tensor::new(rms)?.reshape((batch_size, ()))?,
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct DpoItem {
    pub chosen_x: Vec<u32>,
    pub chosen_y: Vec<u32>,
    pub chosen_mask: Vec<bool>,

    pub rejecten_x: Vec<u32>,
    pub rejecten_y: Vec<u32>,
    pub rejecten_mask: Vec<bool>,
}

pub struct DpoTensorItem {
    pub chosen_x: Tensor<u32>,
    pub chosen_y: Tensor<u32>,
    pub chosen_mask: Tensor<bool>,

    pub rejecten_x: Tensor<u32>,
    pub rejecten_y: Tensor<u32>,
    pub rejecten_mask: Tensor<bool>,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use lumen_core::IndexOp;
    use lumen_dataset::{common::JsonlDataset, Dataset};
    use crate::tokenizer::Tokenizer;

    use crate::dataset::{DpoDataset, DpoDatasetItem};

    use super::DopDataLoader;

    #[test]
    fn test_jsonl_dataset() {
        let dataset = JsonlDataset::<DpoDatasetItem>::new("./assets/cache/dpo.jsonl").unwrap();
        println!("{}", dataset.len());
        println!("{:#?}", dataset.get(10000).unwrap().unwrap());
    }

    #[test]
    fn test_pretrain_dataset() {
        let tokenizer = Tokenizer::from_file("./assets").unwrap();
        let dataset = DpoDataset::new(
            "./assets/cache/dpo.jsonl", Arc::new(tokenizer), 512,
        ).unwrap();

        let item = dataset.get(100).unwrap().unwrap();
        println!("{:?}", &item.chosen_x[..50]);
        println!("{:?}", &item.chosen_y[..50]);
        println!("{:?}", item.chosen_mask);
    }

    #[test]
    fn test_pretrain_dataloader() {
        let tokenizer = Tokenizer::from_file("./assets").unwrap();
        let dataset = DpoDataset::new(
            "./assets/cache/dpo.jsonl", Arc::new(tokenizer), 512,
        ).unwrap();
        let loader = DopDataLoader::from_dataset(dataset, 4, true);
        let batch = loader.iter().next().unwrap().unwrap();
        println!("{}", batch.chosen_x.index((.., ..6)).unwrap());
        println!("{}", batch.chosen_y.index((.., ..6)).unwrap());
    }
}