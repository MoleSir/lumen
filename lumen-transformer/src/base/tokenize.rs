use core::f32;
use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    vocab_inv: HashMap<u32, Vec<u8>>,
    scores: HashMap<u32, f32>,
}

impl Tokenizer {
    pub fn new(vocab_data: Vec<(Vec<u8>, f32)>) -> Self {
        let mut vocab = HashMap::new(); 
        let mut vocab_inv = HashMap::new();
        let mut scores = HashMap::new();

        for (i, (bytes, score)) in vocab_data.into_iter().enumerate() {
            vocab.insert(bytes.clone(), i as u32);
            vocab_inv.insert(i as u32, bytes);
            scores.insert(i as u32, score);
        }

        Self { vocab, vocab_inv, scores }   
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = text.replace(" ", "\u{2581}");

        let mut tokens: Vec<u32> = text
            .bytes()
            .map(|b| *self.vocab.get(&vec![b]).unwrap_or(&0))
            .collect();
        
        loop {
            let mut best_score = -f32::INFINITY;
            let mut best_pair_idx = None;
            let mut best_merged_id = None;

            if tokens.len() < 2 {
                break;
            }

            for i in 0..tokens.len() - 1 {
                let id1 = tokens[i];
                let id2 = tokens[i + 1];

                let mut merged_bytes = self.vocab_inv[&id1].clone();
                merged_bytes.extend_from_slice(&self.vocab_inv[&id2]);

                if let Some(&merged_id) = self.vocab.get(&merged_bytes) {
                    let score = self.scores.get(&merged_id).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_pair_idx = Some(i);
                        best_merged_id = Some(merged_id);
                    }
                }
            }

            if let Some(idx) = best_pair_idx {
                let merged_id = best_merged_id.unwrap();
                tokens[idx] = merged_id;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.vocab_inv.get(&id) {
                bytes.extend_from_slice(token_bytes);
            }
        }

        String::from_utf8_lossy(&bytes).replace("\u{2581}", " ")
    }
}

#[cfg(test)]
mod test {
    use crate::Tokenizer;

    #[test]
    fn simple_vocab() {
        let vocab_data = vec![
            (vec![b'H'], 1.0), 
            (vec![b'e'], 1.0), 
            (vec![b'l'], 1.0), 
            (vec![b'o'], 1.0), 
            (vec![b' '], 1.0), // 普通空格
            (" ".as_bytes().to_vec(), 1.0), // Llama 特殊空格
            // 合并规则
            (vec![b'H', b'e'], 2.0),     // 'He'
            (vec![b'l', b'l'], 3.0),     // 'll' (分数最高，优先合并)
            (vec![b'l', b'o'], 2.0),     // 'lo'
            (vec![b'H', b'e', b'l', b'l', b'o'], 10.0), // 'Hello'
        ];
    
        let tokenizer = Tokenizer::new(vocab_data);
    
        let text = "Hello";
        
        let ids = tokenizer.encode(text);
        println!("Encoded IDs: {:?}", ids);
    
        let decoded = tokenizer.decode(&ids);
        println!("Decoded: {}", decoded);
    }
}