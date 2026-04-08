use std::collections::HashMap;
use crate::types::Token;

use super::{Decode, Model, Normalize, PreTokenize};

pub struct BPEModel {
    pub vocab: HashMap<String, u32>,
    pub merges: HashMap<(String, String), String>,
    pub decoder: HashMap<u32, String>,
}

impl Model for BPEModel {
    type Error = std::io::Error;

    fn tokenize(&self, text: String) -> Result<Vec<Token>, Self::Error> {
        // 初始切分为字符序列
        let mut words: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        loop {
            let mut best_pair = None;

            // 寻找当前序列中 rank 最靠前的 merge 规则
            for i in 0..words.len().saturating_sub(1) {
                let pair = (words[i].clone(), words[i + 1].clone());
                // 这里可以使用 merges 的索引作为优先级
                if let Some(_) = self.merges.get(&pair) {
                    // 在实际实现中，merges 通常存储优先级（rank）
                    // 为了简化，这里假设我们能找到匹配
                    best_pair = Some(pair);
                    break; // 简化版：找到第一个就处理
                }
            }

            if let Some(pair) = best_pair {
                let mut new_words = Vec::new();
                let mut i = 0;
                while i < words.len() {
                    if i < words.len() - 1 && words[i] == pair.0 && words[i+1] == pair.1 {
                        new_words.push(format!("{}{}", pair.0, pair.1));
                        i += 2;
                    } else {
                        new_words.push(words[i].clone());
                        i += 1;
                    }
                }
                words = new_words;
            } else {
                break;
            }
        }

        let tokens = words.iter().map(|w| {
            let id = *self.vocab.get(w).unwrap_or(&0); // 默认 0 为 unk
            Token { id, value: w.clone() }
        }).collect();

        Ok(tokens)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.decoder.get(&id).cloned()
    }
}

pub struct BPETrainer {
    pub vocab_size: usize,
    pub min_frequency: u32,
}

impl BPETrainer {
    pub fn train(&self, corpus: Vec<String>) -> BPEModel {
        let mut word_counts = HashMap::new();
        let b2u = bytes_to_unicode();

        // 1. 将语料转为字节编码后的字符串并统计词频
        for text in corpus {
            let encoded: String = text.as_bytes().iter().map(|b| b2u[b]).collect();
            *word_counts.entry(encoded).or_insert(0u32) += 1;
        }

        // 2. 初始化词表（单字节字符）
        let mut vocab: HashMap<String, u32> = b2u.values()
            .enumerate()
            .map(|(i, &c)| (c.to_string(), i as u32))
            .collect();
        
        let mut merges = HashMap::new();
        let mut current_vocab_size = vocab.len();

        // 3. 迭代合并
        while current_vocab_size < self.vocab_size {
            let mut pair_counts = HashMap::new();
            for (word, freq) in &word_counts {
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect(); // 这里简化了，实际应记录状态
                for i in 0..chars.len().saturating_sub(1) {
                    let pair = (chars[i].clone(), chars[i+1].clone());
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }

            if let Some((best_pair, freq)) = pair_counts.into_iter().max_by_key(|&(_, count)| count) {
                if freq < self.min_frequency { break; }

                let new_token = format!("{}{}", best_pair.0, best_pair.1);
                vocab.insert(new_token.clone(), current_vocab_size as u32);
                merges.insert(best_pair, new_token.clone());
                
                // 更新 word_counts (省略具体实现，逻辑是把 word 里的 pair 替换为 new_token)
                current_vocab_size += 1;
            } else {
                break;
            }
        }

        let decoder = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        BPEModel { vocab, merges, decoder }
    }
}

// --- Normalizer ---
pub struct ByteNormalizer;
impl Normalize for ByteNormalizer {
    type Error = std::io::Error;
    fn normalize(&self, text: String) -> Result<String, Self::Error> {
        Ok(text) // BBPE 通常不强制小写，保留原始字节信息
    }
}

// --- PreTokenizer (GPT-2 Style Regex) ---
use regex::Regex;
pub struct GPT2PreTokenizer {
    re: Regex,
    byte_map: HashMap<u8, char>,
}

impl GPT2PreTokenizer {
    pub fn new() -> Self {
        Self {
            re: Regex::new(r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#).unwrap(),
            byte_map: bytes_to_unicode(),
        }
    }
}

impl PreTokenize for GPT2PreTokenizer {
    type Error = std::io::Error;
    fn pre_tokenize(&self, text: String) -> Result<Vec<String>, Self::Error> {
        let mut res = vec![];
        for mat in self.re.find_iter(&text) {
            // 将片段转为字节映射后的可见字符
            let b_str: String = mat.as_str().as_bytes().iter().map(|b| self.byte_map[b]).collect();
            res.push(b_str);
        }
        Ok(res)
    }
}

// --- Decoder ---
pub struct BPEDecoder {
    reverse_byte_map: HashMap<char, u8>,
}

impl Decode for BPEDecoder {
    type Error = std::io::Error;
    fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error> {
        let combined = tokens.join("");
        let bytes: Vec<u8> = combined.chars()
            .map(|c| *self.reverse_byte_map.get(&c).unwrap_or(&0))
            .collect();
        
        String::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

pub fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~')
        .chain(b'\xA1'..=b'\xAC')
        .chain(b'\xAE'..=b'\xFF')
        .collect();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs.into_iter().map(|c| std::char::from_u32(c).unwrap()))
        .collect()
}