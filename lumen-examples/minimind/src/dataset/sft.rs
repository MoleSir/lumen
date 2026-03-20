// use std::path::Path;

// use anyhow::Context;
// use lumen_dataset::{common::JsonlDataset, Dataset};
// use serde::Deserialize;
// use tokenizers::Tokenizer;

// pub struct SftDataset {
//     pub tokenizer: Tokenizer,
//     pub max_length: usize,
//     bos_id: u32, 
//     eos_id: u32,
//     pad_token_id: u32,
//     jsonl_dataset: JsonlDataset<SftItem>,
// }

// impl SftDataset {
//     pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Tokenizer, max_length: usize) -> anyhow::Result<Self> {
//         let jsonl_dataset = JsonlDataset::new(data_path).context("new jsonl dataset")?;
//         let bos_id = tokenizer.token_to_id("<|im_start|>").expect("BOS token not found");
//         let eos_id = tokenizer.token_to_id("<|im_end|>").expect("EOS token not found");
//         let pad_token_id = tokenizer.token_to_id("<|endoftext|>").expect("PAD token not found");

//         Ok(Self {
//             tokenizer,
//             max_length,
//             bos_id,
//             eos_id,
//             pad_token_id,
//             jsonl_dataset,
//         })
//     }

//     fn create_chat_prompt(&self, conversations: &[Message]) -> anyhow::Result<()> {
//         let tools = if !conversations.is_empty() && conversations[0].role == Role::System && conversations[0].functions.is_some() {
//             Some(conversations[0].functions.as_ref().unwrap())
//         } else {
//             None
//         };

//         Ok(())
//     }    
// }

// impl Dataset for SftDataset {
//     type Error = anyhow::Error;

//     fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error> {
//         let item = self.jsonl_dataset.get(index)?;
//         let item = match item {
//             Some(item) => item,
//             None => return Ok(None),
//         };

        
//     }

//     fn len(&self) -> usize {
//         self.jsonl_dataset.len()
//     }
// }

// #[derive(Debug, Deserialize)]
// pub struct SftItem {
//     pub conversations: Vec<Message>,
// }

// #[derive(Debug, Deserialize, PartialEq, Eq)]
// #[serde(rename_all = "lowercase")]
// pub enum Role {
//     System,
//     User,
//     Assistant,
// }

// #[derive(Debug, Deserialize)]
// pub struct Message {
//     pub role: Role,
//     pub content: String,

//     #[serde(default)]
//     pub functions: Option<serde_json::Value>,
// }

// // #[derive(Debug, Deserialize)]
// // pub struct Function {
// //     pub name: String,
// //     #[serde(default)]
// //     pub description: Option<String>,
// //     #[serde(default)]
// //     pub parameters: Option<serde_json::Value>,
// // }