use std::path::Path;
use anyhow::Context;
use lumen_core::Tensor;
use lumen_dataset::{common::JsonlDataset, Dataset};
use serde::Deserialize;
use crate::{model::IGNORE_ID, tokenizer::{EncodeOptions, Tokenizer}};
use super::types::Message;

pub struct SftDataset {
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    pub assistant_bos_ids: Vec<u32>,
    pub assistant_eos_ids: Vec<u32>,
    jsonl_dataset: JsonlDataset<SftItem>,
}

impl SftDataset {
    pub fn new<P: AsRef<Path>>(data_path: P, tokenizer: Tokenizer, max_length: usize) -> anyhow::Result<Self> {
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

    pub fn create_chat_prompt(&self, conversations: &[Message]) -> String {
        self.tokenizer.apply_chat_template(conversations, false)
    }

    /// 给原始对话进行数据增强（随机增加 system prompt）
    fn pre_processing_chat(conversations: Vec<Message>) -> Vec<Message> {
        const ADD_SYSTEM_RATIO: f64 = 0.2;
        const SYSTEM_PROMPTS: [&'static str; 10] = [
            "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
            "你是minimind，一个小巧但有用的语言模型。",
            "你是一个专业的AI助手，请提供有价值的回答。",
            "你是minimind，请尽力帮助用户解决问题。",
            "你是一个可靠的AI，请给出准确的回答。",
            "You are a helpful AI assistant.",
            "You are minimind, a lightweight intelligent assistant.",
            "You are a friendly chatbot. Please answer the user's questions carefully.",
            "You are a knowledgeable AI. Try your best to provide accurate information.",
            "You are minimind, a small but useful language model."
        ];

        if conversations.len() == 0 {
            return vec![];
        }

        if conversations[0].is_system() {
            conversations
        } else {
            let ratio = rand::random_range(0.0..1.0);
            if ratio < ADD_SYSTEM_RATIO {
                let index = rand::random_range(0..SYSTEM_PROMPTS.len());
                let mut new_conversations = vec![Message::system(SYSTEM_PROMPTS[index], [])];
                new_conversations.extend(conversations);
                new_conversations
            } else {    
                conversations
            }
        }
    }

    // 找到 input_ids 序列中由 assistant_bos_ids 和 assistant_eos_ids 包裹的部分，使得其有效
    fn generate_labels(&self, input_ids: &[u32]) -> Vec<u32> {
        let mut labels = vec![IGNORE_ID; input_ids.len()];
        let bos = &self.assistant_bos_ids;
        let eos = &self.assistant_eos_ids;

        let mut i = 0;
        while i + bos.len() < input_ids.len() {
            // 匹配 Assistant 回复的开始
            if input_ids[i..].starts_with(bos) {
                let content_start = i + bos.len();
                let mut found_eos = false;
                
                // 从开始位置寻找对应的结束符
                let mut j = content_start;
                while j + eos.len() <= input_ids.len() {
                    if input_ids[j..].starts_with(eos) {
                        let content_end = j + eos.len();
                        
                        // 填充 Label 区域
                        // 这里的逻辑是：如果是因果语言模型，训练目标是输入 input[k]，预测 input[k+1]
                        // 所以 labels[k] = input_ids[k+1]
                        for k in i..content_end - 1 {
                            if k < input_ids.len() - 1 {
                                labels[k] = input_ids[k + 1];
                            }
                        }
                        
                        i = content_end;
                        found_eos = true;
                        break;
                    }
                    j += 1;
                }
                if !found_eos { break; }
            } else {
                i += 1;
            }
        }
        labels
    }
}

#[derive(Debug, Deserialize)]
pub struct SftItem {
    pub conversations: Vec<Message>,
}

impl Dataset for SftDataset {
    type Error = anyhow::Error;
    type Item = (Tensor<u32>, Tensor<u32>);

    fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error> {
        let v = self.jsonl_dataset.get(index)?;
        let v = match v {
            Some(v) => v,
            None => return Ok(None),
        };

        // 数据增强
        let conversations = Self::pre_processing_chat(v.conversations);

        // 创建 chat template
        let prompt = self.create_chat_prompt(&conversations);
        println!("{}", prompt);

        // 转为 tokens
        let encoding = self.tokenizer.encode(&prompt, EncodeOptions::default())?;
        // 填充 pad token（先一定填充一个，之后再补偿）
        let mut input_ids = encoding.get_ids().to_vec();
        input_ids.push(self.tokenizer.pad_token_id());
        if input_ids.len() < self.max_length {
            for _ in 0..self.max_length - input_ids.len() {
                input_ids.push(self.tokenizer.pad_token_id());
            } 
        } 

        // 生成 labels
        let labels = self.generate_labels(&input_ids);

        assert_eq!(input_ids.len(), labels.len());

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
    use crate::tokenizer::Tokenizer;

    use crate::dataset::{SftDataset, SftItem};

    #[test]
    fn test_jsonl_dataset() {
        let dataset = JsonlDataset::<SftItem>::new("./assets/cache/sft_mini_512.jsonl").unwrap();
        println!("{}", dataset.len());
        println!("{:#?}", dataset.get(10000).unwrap().unwrap().conversations);
    }

    #[test]
    fn test_pretrain_dataset() {
        let tokenizer = Tokenizer::from_file("./assets").unwrap();
        let dataset = SftDataset::new(
            "./assets/cache/sft_mini_512.jsonl", tokenizer, 512,
        ).unwrap();

        let (input_ids, labels) = dataset.get(0).unwrap().unwrap();
        println!("input_ids: {}\n\n", input_ids);
        println!("labels: {}\n\n", labels);
    }
}