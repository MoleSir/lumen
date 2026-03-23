use std::{path::Path, sync::Arc};

use anyhow::Context;
use lumen_dataset::{common::JsonlDataset, Dataset};
use serde::Deserialize;
use crate::tokenizer::Tokenizer;
use super::types::Message;

pub struct RLAIDataset {
    tokenizer: Arc<Tokenizer>,
    jsonl_dataset: JsonlDataset<RLAIItem>,
}

#[derive(Debug, Deserialize)]
pub struct RLAIItem {
    pub conversations: Vec<Message>,
}

impl RLAIDataset {
    pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Arc<Tokenizer>) -> anyhow::Result<Self> {
        let jsonl_dataset = JsonlDataset::new(data_path).context("new jsonl dataset")?;

        // // assistant 回复的开始/结束序列：<bos>assistant\n"xxxxx"<eos>\n
        // let assistant_bos_ids = tokenizer
        //     .encode(&format!("{}assistant\n", tokenizer.bos_token()), EncodeOptions::default())?
        //     .get_ids()
        //     .to_vec();
        // let assistant_eos_ids = tokenizer
        //     .encode(&format!("{}\n", tokenizer.eos_token()), EncodeOptions::default())?
        //     .get_ids()
        //     .to_vec();

        Ok(Self {
            tokenizer,
            jsonl_dataset,
        })
    }

    // conversations 应该是 system / user 对，其最后一个 user 的 content 为 answer，之前的所有 messages 为 conversations
    pub fn create_chat_prompt(&self, mut conversations: Vec<Message>) -> anyhow::Result<(String, String)> {
        let message = conversations.pop().ok_or_else(|| anyhow::anyhow!("no message!"))?;
        let answer = if let Message::User(msg) = message {
            msg.content
        } else {
            return Err(anyhow::anyhow!("should end with a user messages!"))
        };

        let prompt = self.tokenizer.apply_chat_template(&conversations, true);

        Ok((prompt, answer))
    }
}


impl Dataset for RLAIDataset {
    type Error = anyhow::Error;
    type Item = (String, String);

    fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error> {
        let v = self.jsonl_dataset.get(index)?;
        let v = match v {
            Some(v) => v,
            None => return Ok(None),
        };

        let result = self.create_chat_prompt(v.conversations)?;
        Ok(Some(result))
    }

    fn len(&self) -> usize {
        self.jsonl_dataset.len()
    }
}