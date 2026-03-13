use std::{collections::{HashMap, HashSet}, u32};
use crate::{components::Model, types::{PreToken, Token}};

#[derive(Debug, Clone)]
pub struct BPEModel {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    merges: HashMap<(String, String), u32>,
    unk_token: String,
}

impl BPEModel {
    pub fn new(
        vocab: HashMap<String, u32>, 
        merges: HashMap<(String, String), u32>,
        unk_token: String
    ) -> Self {
        let vocab_r = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self { vocab, vocab_r, merges, unk_token }
    }

    pub fn merge_step(&self, word: &Vec<String>) -> (Vec<String>, bool) {
        if word.len() < 2 {
            return (word.clone(), false);
        }

        let mut min_rank = u32::MAX;
        let mut best_pair = None;

        // 1. 寻找当前 word 中 rank 最小（优先级最高）的相邻对
        for i in 0..word.len() - 1 {
            let pair = (word[i].clone(), word[i+1].clone());
            if let Some(&rank) = self.merges.get(&pair) {
                if rank < min_rank {
                    min_rank = rank;
                    best_pair = Some(pair);
                }
            }
        }

        // 2. 如果找到了可合并的对，进行合并
        if let Some(best_pair) = best_pair {
            let mut new_word = Vec::new();
            let mut i = 0;
            let mut changed = false;

            while i < word.len() {
                if i < word.len() - 1 
                    && word[i] == best_pair.0 
                    && word[i+1] == best_pair.1 
                {
                    new_word.push(format!("{}{}", word[i], word[i+1]));
                    i += 2;
                    changed = true;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            return (new_word, changed)
        }

        (word.clone(), false)
    }
}

impl Model for BPEModel {
    type Error = BPEError;
    
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn tokenize(&self, pre_tokens: Vec<PreToken>) -> Result<Vec<Token>, Self::Error> {
        let mut tokens = vec![];
        let unk_id = *self.vocab
            .get(&self.unk_token)
            .ok_or_else(|| BPEError::UNKTokenNotFound(self.unk_token.clone()))?;

        for pt in pre_tokens {
            // 对每个 pre_tokens，将其中的 value(String) 进行合并
            let mut word_parts: Vec<String> = pt.value.chars().map(|c| c.to_string()).collect();
            loop {
                let (new_parts, changed) = self.merge_step(&word_parts);
                word_parts = new_parts;
                if !changed {
                    break;
                }
            }

            // 此时 value 被拆分为 Vec<String>，映射到 token id 
            let mut current_offsets = pt.offset.0;
            for part in word_parts {
                let part_len = part.len();
                let id = self.vocab.get(&part).copied().unwrap_or(unk_id);
                tokens.push(Token::new(id, part, (current_offsets, current_offsets + part_len)));
                current_offsets += part_len;
            }
        }

        Ok(tokens)
    }   
}

pub struct BPETrainer {
    vocab_size: usize,
    special_tokens: Vec<String>,
}

impl BPETrainer {
    pub fn new(vocab_size: usize, special_tokens: Vec<String>) -> Self {
        Self { vocab_size, special_tokens }
    }

    pub fn train(&self, text: &str) -> Result<BPEModel, BPEError> {
        // 1. 预处理：统计单词频率
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            *word_counts.entry(word.to_string()).or_insert(0) +=1 ;
        }

        // 2. 拆分为字符列表
        let mut splits: HashMap<Vec<String>, usize> = word_counts.iter()
            .map(|(word, count)| {
                let chars = word.chars().map(|c| c.to_string()).collect::<Vec<_>>();
                (chars, *count)
            })
            .collect();

        // 3. 初始化词表 (包含语料中出现的所有基础字符 + 特殊 token)
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        let mut next_id = 0;
        // 加入特殊 token
        for t in &self.special_tokens {
            vocab.insert(t.clone(), next_id);
            next_id += 1;
        }
        // 基础字符
        let mut alphabet: HashSet<String> = HashSet::new();
        for word_chars in splits.keys() {
            for char_s in word_chars {
                alphabet.insert(char_s.clone());
            }
        }
        for char_s in alphabet {
            if !vocab.contains_key(&char_s) {
                vocab.insert(char_s, next_id);
                next_id += 1;
            }
        }

        // 4. 循环，不断找到最高频的 Pair 合并
        while vocab.len() < self.vocab_size {
            // 统计所有相邻 Pair 的次数
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (word_parts, count) in splits.iter() {
                if word_parts.len() < 2 {
                    continue;
                }
                for i in 0..word_parts.len() - 1 {
                    let pair = (word_parts[i].clone(), word_parts[i+1].clone());
                    *pair_counts.entry(pair).or_insert(0) += count;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // 找出频率最高的 Pair
            let best_pair = pair_counts.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(p, _)| p.clone())
                .unwrap();

            // 新 Token
            let new_token = format!("{}{}", best_pair.0, best_pair.1);

            // println!("Merging: {:?} -> {:?} (vocab size: {})", best_pair, new_token, vocab.len() + 1);
            
            // 更新记录
            vocab.insert(new_token.clone(), next_id);
            merges.insert(best_pair.clone(), next_id);

            next_id += 1;

            // 更新 splits
            let mut new_splits = HashMap::new();
            for (word_parts, count) in splits {
                let mut new_word_parts = Vec::new();
                let mut i = 0;
                while i < word_parts.len() {
                    if i < word_parts.len() - 1 && word_parts[i] == best_pair.0 && word_parts[i+1] == best_pair.1 {
                        new_word_parts.push(new_token.clone());
                        i += 2;
                    } else {
                        new_word_parts.push(word_parts[i].clone());
                        i += 1;
                    }
                }
                new_splits.insert(new_word_parts, count);
            }
            splits = new_splits
        }

        let unk = self.special_tokens.first().cloned().unwrap_or_else(|| "<UNK>".to_string());

        Ok(BPEModel::new(vocab, merges, unk))
        
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BPEError {
    #[error("UNK token {0} not found in vocab")]
    UNKTokenNotFound(String)
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use crate::{components::{model::BPETrainer, pre_tokenize::WhitespacePreTokenizer, Model}, tokenizer::Tokenizer};
    use super::BPEModel;

    #[test]
    fn test_bpemodel() {
        let mut vocab = HashMap::new();
        vocab.insert("<UNK>".to_string(), 0);
        vocab.insert("h".to_string(), 1);
        vocab.insert("e".to_string(), 2);
        vocab.insert("l".to_string(), 3);
        vocab.insert("o".to_string(), 4);
        vocab.insert("w".to_string(), 5);
        vocab.insert("r".to_string(), 6);
        vocab.insert("d".to_string(), 7);
        vocab.insert("el".to_string(), 8);  // Merge result
        vocab.insert("lo".to_string(), 9);  // Merge result
        vocab.insert("hello".to_string(), 10); // Merge result
    
        let mut merges = HashMap::new();
        merges.insert(("e".to_string(), "l".to_string()), 1); // "e" + "l" -> "el" (Rank 1)
        merges.insert(("l".to_string(), "l".to_string()), 2); // "l" + "l" -> "ll" (Rank 2) - 假设
        merges.insert(("l".to_string(), "o".to_string()), 3); // "l" + "o" -> "lo" (Rank 3)
        merges.insert(("h".to_string(), "el".to_string()), 4); // "h" + "el" -> "hel"
        merges.insert(("hel".to_string(), "lo".to_string()), 5); // "hel" + "lo" -> "hello"

        let bpe_model = BPEModel::new(vocab, merges, "<UNK>".to_string());
        
        let tokenizer = Tokenizer::new_model(bpe_model)
            .with_pre_tokenize(WhitespacePreTokenizer);

        let text = "hello world";
        println!("Input: {}", text);
        
        let tokens = tokenizer.encode(text).expect("encode");
        println!("Encoded Tokens:");
        for t in &tokens {
            println!("ID: {}, Value: {:?}, Offset: {:?}", t.id, t.value, t.offset);
        }
        
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let decoded = tokenizer.decode(&ids).expect("decode");
        println!("Decoded: {}", decoded);
    }

    #[test]
    fn test_train() {
        let text = "hug ".repeat(10) + "pug ".repeat(5).as_str() + "pun ".repeat(12).as_str() + "bun ".repeat(4).as_str();
        println!("{}", text);

        let trainer = BPETrainer::new(20, vec!["<UNK>".to_string()]);

        println!("--- 开始训练 ---");
        let model = trainer.train(&text).unwrap();
        println!("--- 训练完成 ---");

        println!("\nModel Merges: {:?}", model.merges);
        println!("Model Vocab Size: {}", model.vocab.len());

        let test_str = "bun bug";
        println!("\n--- Tokenizing: '{}' ---", test_str);

        match model.tokenize(vec![test_str.into()]) {
            Ok(tokens) => {
                for t in tokens {
                    println!("Token: {:?} (ID: {})", t.value, t.id);
                }
            },
            Err(e) => println!("Error: {}", e),
        }

        let test_unk = "hugs@"; // '@' 不在训练数据中
        println!("\n--- Tokenizing UNK case: '{}' ---", test_unk);
        match model.tokenize(vec![test_unk.into()]) {
            Ok(tokens) => {
                for t in tokens {
                    println!("Token: {:?} (ID: {})", t.value, t.id);
                }
            },
            Err(e) => println!("Error: {}", e),
        }
    }
}