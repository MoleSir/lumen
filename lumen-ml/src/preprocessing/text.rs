use std::collections::{HashMap, HashSet};
use lumen_core::Tensor;

pub struct CountVectorizer {

}

impl CountVectorizer {
    pub fn transform(&self, texts: &[&str]) -> lumen_core::Result<Tensor<u32>> {
        let mut vocab = HashSet::new();
        for text in texts {
            for token in text.split_whitespace() {
                vocab.insert(token);
            }
        } 

        let vocab = vocab.into_iter()
            .enumerate().map(|(i, t)| (t, i)).collect::<HashMap<_, _>>();

        let mut output = vec![];
        for text in texts {
            let mut counter = vec![0u32; vocab.len()];
            for token in text.split_whitespace() {
                let index = vocab[token];
                counter[index] += 1;
            }
            output.push(Tensor::new(counter)?); 
        }

        // [(vocab.len(),); n_samples] => (n_samples, vocab.len())
        let output = Tensor::stack(&output, 0)?;
            
        Ok(output)
    }
}
