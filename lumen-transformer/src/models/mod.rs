use std::path::PathBuf;
use async_stream::stream;
use futures_core::Stream;
use lumen_core::{FloatDType, IndexOp, IntTensor, Tensor};

use crate::Sampler;

pub mod llama;
pub mod gpt2;
pub mod deepseek;
pub mod qwen2;
pub mod common;

pub trait PretrainedModel<T: FloatDType> : Sized {
    type Error;
    fn from_pretrained(path: impl Into<PathBuf>) -> Result<Self, Self::Error>;
}

pub trait ForCausalLM<T: FloatDType> {
    type Cache;
    type Error: From<lumen_core::Error>;

    fn new_cache(&self) -> Result<Self::Cache, Self::Error>;

    fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut Self::Cache) -> Result<Tensor<T>, Self::Error>;

    fn generate(&self, input_ids: &[u32], max_new_tokens: usize, eos_token: Option<u32>, sample: &Sampler) -> Result<Vec<u32>,Self::Error> {
        let mut cache = self.new_cache()?;
        let mut generated = vec![];

        // 1. Prefill: (1, seq_len)
        let mut input_tensor = Tensor::new(input_ids)?.unsqueeze(0)?;
        let mut start_pos = 0;

        for _ in 0..max_new_tokens {
            // (1, seq_len, vocab_size)
            let seq_len = input_tensor.dims()[1];
            let logits = self.forward(&input_tensor, start_pos, &mut cache)?;
            
            // 取最后一个 token 的 logits
            // (1, seq_len, vocab_size) -> (seq_len, vocab_size) -> (vocab_size)
            let next_token_logits = logits.squeeze(0)?.index((seq_len-1, ..))?;
            
            // 2. 采样阶段 
            let next_token_id = sample.sample(&next_token_logits);
            if let Some(eos_token) = eos_token {
                if eos_token == next_token_id {
                    break;
                }
            }
            generated.push(next_token_id);
            
            // 3. 准备下一步 Decode 的输入 (只需要输入这一个新生成的 Token)
            start_pos += seq_len;
            input_tensor = Tensor::new(next_token_id)?.reshape((1, 1))?;
        }

        Ok(generated)
    }

    fn generate_stream(
        &self, input_ids: &[u32], max_new_tokens: usize, eos_token: Option<u32>, sample: &Sampler
    ) -> impl Stream<Item = Result<u32, Self::Error>> {
        stream! {
            let mut cache = self.new_cache()?;
            let mut input_tensor = Tensor::new(input_ids)?.unsqueeze(0)?;
            let mut start_pos = 0;
    
            for _ in 0..max_new_tokens {
                let seq_len = input_tensor.dims()[1];
                let logits = self.forward(&input_tensor, start_pos, &mut cache)?;
    
                let next_logits = logits.squeeze(0)?.index((seq_len - 1, ..))?;
                let next_token = sample.sample(&next_logits);
                if let Some(eos_token) = eos_token {
                    if eos_token == next_token {
                        break;
                    }
                }
    
                yield Ok(next_token);
    
                start_pos += seq_len;
                input_tensor = Tensor::new(next_token)?.reshape((1, 1))?;
            }
        }
    }
} 