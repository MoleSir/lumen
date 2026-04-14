use std::collections::HashMap;
use fancy_regex::Regex as FancyRegex;
use regex::Regex;

pub struct ByteLevelBPE {
    pub byte_encoder: HashMap<u8, char>,
    pub byte_decoder: HashMap<char, u8>,
    pub vocab: HashMap<String, u32>,
    pub decoder: HashMap<u32, String>,
    pub bpe_ranks: HashMap<(String, String), u32>,
    pub pat: FancyRegex,
    pub added_tokens: Vec<String>,
    pub added_tokens_vocab: HashMap<String, u32>,
    pub added_regex: Option<Regex>,
}

impl ByteLevelBPE {
    pub fn new(added_tokens: Vec<String>) -> Self {
        // 1. 基础映射表：Byte(0~255) -> 可见 Unicode 字符
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<_, _> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

        // 2. 词表 (vocab): token_str -> token_id
        let vocab: HashMap<String, u32> = byte_encoder
            .values()
            .enumerate()
            .map(|(i, c)| (format!("{}", c), i as u32))
            .collect();
        let decoder: HashMap<u32, String> = vocab
            .iter()
            .map(|(s, i)| (*i, s.clone()))
            .collect();
        
        // 3. BPE 合并规则优先级 (merges)
        let bpe_ranks = HashMap::new();

        // 4. 预分词正则表达式 (照搬 GPT-2/3 的官方正则)
        let pat_str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        let pat = FancyRegex::new(pat_str).expect("Invalid GPT-2 pattern");

        // 5. 提前编译 added Tokens 正则，用于 Train 和 Encode 阶段的切分隔离
        let added_tokens_vocab = HashMap::new();
        let added_regex = if added_tokens.is_empty() {
            None
        } else {
            let escaped_tokens: Vec<String> = added_tokens
                .iter()
                .map(|t| regex::escape(t))
                .collect();
            
            // 组合成 (token1|token2|token3)
            let pattern = format!("({})", escaped_tokens.join("|"));
            Some(Regex::new(&pattern).expect("Invalid added tokens pattern"))
        };

        Self {
            byte_decoder,
            byte_encoder,
            vocab,
            decoder,
            bpe_ranks,
            pat,
            added_tokens,
            added_tokens_vocab,
            added_regex,
        }
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) {
        assert!(vocab_size > 256);
        let num_merges = vocab_size - 256;

        // 1. 按 Added Tokens 隔离切分，防止污染 BPE 统计
        let chunks = self.split_input_text_by_added_tokens(text);

        // 2. 对非特殊 token 的纯净文本进行预分词
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for chunk in chunks {
            // 空字符或者特殊 token 绝对不计入统计！
            if chunk.is_empty() || self.added_tokens.iter().any(|t| t == chunk) {
                continue;
            }
            
            // 使用 pat 对每个 chunk 进行切分
            for mat_res in self.pat.find_iter(chunk) {
                if let Ok(mat) = mat_res {
                    let word = mat.as_str().to_string();
                    // 计数更新
                    *word_counts.entry(word).or_insert(0) += 1;
                }
            }
        }

        // 3. 将单词转化为基础的 byte-level unicode 元组
        let mut bpe_vocab = HashMap::new();
        for (word, count) in word_counts {
            // 将这个 word 转为 Vec<u8> 后通过 byte_encoder 得到 Vec<char>
            let byte_word: Vec<String> = word.as_bytes().iter().map(|b| format!("{}", self.byte_encoder[b])).collect();
            bpe_vocab.insert(byte_word, count);
        }

        // 4. BPE 核心训练循环
        for i in 0..num_merges {
            let pairs = Self::get_stats(&bpe_vocab);
            if pairs.is_empty() {
                break;
            }

            let best_pair = pairs.iter().max_by_key(|t| t.1).unwrap().0;
            self.bpe_ranks.insert(best_pair.clone(), i as u32);

            // 更新词表
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            let new_id = self.vocab.len() as u32;
            self.vocab.insert(new_token.clone(), new_id);
            self.decoder.insert(new_id, new_token);

            // 合并语料中的 pair
            bpe_vocab = Self::merge_vocab(best_pair, &bpe_vocab); 
            
        }

        // 5. 训练结束，给 Special Tokens 分配词表尾部的独立 ID
        let current_max_id = *self.vocab.values().max().unwrap();
        for (i, added) in self.added_tokens.iter().enumerate() {
            let st_id = current_max_id + 1 + i as u32;
            self.added_tokens_vocab.insert(added.clone(), st_id);
            self.decoder.insert(st_id, added.clone());
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut bpe_ids = vec![];

        // 1. 按 Added Tokens 隔离切分
        let chunks = self.split_input_text_by_added_tokens(text);

        for chunk in chunks {
            if chunk.is_empty() {
                continue;
            }

            // 2. 如果这部分是特殊 token，直接查 special_vocab 转 ID
            if self.added_tokens.iter().any(|t| t == chunk) {
                bpe_ids.push(self.added_tokens_vocab[chunk]);
                continue
            }

            // 3. 如果是普通文本，走 BBPE 切分逻辑
            for mat_res in self.pat.find_iter(chunk) {
                if let Ok(mat) = mat_res {
                    let word = mat.as_str();
                    // 3.1 字符串 -> utf8 bytes -> unicode 可见字符序列
                    let byte_word: Vec<String> = word.as_bytes().iter().map(|b| self.byte_encoder[b].to_string()).collect();
                    
                    // 3.2 根据 rank 推断 BPE tokens
                    let bpe_tokens = self.get_bpe_word(byte_word);

                    // 3.3 Token -> ID
                    for bpe_token in bpe_tokens {
                        bpe_ids.push(self.vocab[&bpe_token]);
                    }
                }
            }
        }

        bpe_ids
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let mut text_bytes: Vec<u8> = vec![];
        let mut texts = vec![];
    
        for &tid in token_ids {
            let token_str = self.decoder.get(&tid).expect("Unknown token id");
            
            if self.added_tokens_vocab.contains_key(token_str) {
                // 遇到特殊 token 前，先把累积的普通 bytes 解码
                if !text_bytes.is_empty() {
                    texts.push(String::from_utf8(text_bytes).unwrap());
                    text_bytes = vec![];
                }
                texts.push(token_str.clone());
            } else {
                for c in token_str.chars() {
                    text_bytes.push(self.byte_decoder[&c]);
                }
            }
        }

        if !text_bytes.is_empty() {
            texts.push(String::from_utf8(text_bytes).unwrap());
        }

        texts.join("")
    }
}

impl ByteLevelBPE {
    /// 统计词表中所有相邻 token pair 的出现频次
    fn get_stats(vocab: &HashMap<Vec<String>, usize>) -> HashMap<(String, String), usize> {
        let mut pairs = HashMap::new();
        for (word, freq) in vocab {
            for i in 0..word.len().saturating_sub(1) {
                let pair = (word[i].clone(), word[i+1].clone());
                *pairs.entry(pair).or_default() += freq;
            }
        }
        pairs
    }

    /// 将语料(v_in)中出现了 pair 的地方合并成一个新的 token
    fn merge_vocab(pair: &(String, String), vocab_in: &HashMap<Vec<String>, usize>) -> HashMap<Vec<String>, usize> {
        let mut vocab_out = HashMap::with_capacity(vocab_in.len());
        let (first, second) = pair;
        let replacement = format!("{}{}", first, second);
    
        for (word_vec, &freq) in vocab_in {
            let mut new_word = Vec::new();
            let mut i = 0;
    
            while i < word_vec.len() {
                // 如果当前项和下一项匹配 pair
                if i < word_vec.len() - 1 && &word_vec[i] == first && &word_vec[i+1] == second {
                    new_word.push(replacement.clone());
                    i += 2; // 跳过两个，因为它们合并了
                } else {
                    new_word.push(word_vec[i].clone());
                    i += 1;
                }
            }
            vocab_out.insert(new_word, freq);
        }
    
        vocab_out
    }

    /// TODO: 利用正则实现？
    fn split_input_text_by_added_tokens<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        let mut result = Vec::new();
    
        if let Some(r) = &self.added_regex {
            let mut last = 0;
            for m in r.find_iter(text) {
                if m.start() > last {
                    result.push(&text[last..m.start()]);
                }
                result.push(m.as_str()); // 保留 token
                last = m.end();
            }
            if last < text.len() {
                result.push(&text[last..]);
            }
        } else {
            result.push(text);
        }
    
        result
    }

    /// 给定字节序列，严格按照训练时的 rank 优先级进行合并。不能从左向右合并，必须从全局 rank 最小的开始！
    fn get_bpe_word(&self, mut words: Vec<String>) -> Vec<String> {
        while words.len() > 1 {
            let pairs: Vec<_> = (0..words.len()-1)
                .into_iter() 
                .map(|i| (words[i].clone(), words[i+1].clone()))
                .collect();

            let best_pair = pairs.iter()
                .map(|p| (p, self.bpe_ranks.get(p)))
                .filter(|(p, r)| r.is_some())
                .map(|(p, r)| (p, r.unwrap()))
                .min_by_key(|pr| pr.1);
            
            match best_pair {
                Some(best_pair) => {
                    let (first, second) = best_pair.0;
                    let mut new_words = vec![];
                    let mut i = 0;
                    while i < words.len() {
                        if i < words.len() - 1 && words[i].as_str() == first.as_str() && words[i+1].as_str() == second.as_str() {
                            new_words.push(format!("{}{}", first, second));
                            i += 2;
                        } else {
                            new_words.push(words[i].clone());
                            i += 1;
                        }
                    }
                    words = new_words;
                }
                None => break // 没有任何可合并的 pair，退出
            }
        }   

        words
    }
}

/// 将 0-255 的 bytes 映射到可见 Unicode，
/// 如果原来的 byte 值就是可见 ascii，映射的就是这个 ascii，否则映射到一个 > 256 的 Unicode，避免 
pub fn bytes_to_unicode() -> HashMap<u8, char> {
    // 可见部分
    let mut bs: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    
    // 遍历 0 到 255 所有 u8 值
    let mut n = 0u32;
    for b in 0..=255 {
        // 遍历到的 b 不在 bs 说明这个值对应的 ascii 不可见
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(std::char::from_u32(256 + n).expect("char from u32"));
            n += 1;
        }
    }

    // 创建 byte 到 char 的映射
    bs.into_iter().zip(cs.into_iter()).collect()
}

#[cfg(test)]
mod tests {
    use super::{bytes_to_unicode, ByteLevelBPE};

    #[test]
    fn test_bytes_to_unicode() {
        let byte_encoder = bytes_to_unicode();
        assert_eq!(byte_encoder.len(), 256);
        for (b, c) in byte_encoder {
            println!("{b}: {c}");
        }
    }

    #[test]
    fn test_added_tokens() {
        let bpe = ByteLevelBPE::new(vec!["<eos>".to_string()]);
        for chunk in bpe.split_input_text_by_added_tokens("abs<eos>dasda") {
            println!("{}", chunk);
        }
    }

    #[test]
    fn test_train() {
        let corpus = "Hello world! This is an LLM BPE implementation example. AI is shaping the world's future.\n测试一下中文能否正常切分。<pad>";

        let mut tokenizer = ByteLevelBPE::new(vec![
            "<|endoftext|>".to_string(),
            "[MASK]".to_string(),
            "<pad>".to_string(),
        ]);
        tokenizer.train(corpus, 356);

        let test_text = "Hello [MASK] future! 这是一个BPE测试。<|endoftext|>";
        let encoded_ids = tokenizer.encode(test_text);
        println!(">>> [Encode 结果 (Token IDs)]:\n{encoded_ids:?}\n");

        let decoded_text = tokenizer.decode(&encoded_ids);
        println!(">>> [Decode 结果 (还原文本)]:\n{decoded_text}\n");
            
        assert_eq!(test_text, decoded_text);
    }
}
