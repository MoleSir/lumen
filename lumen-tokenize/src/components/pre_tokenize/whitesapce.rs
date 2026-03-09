use std::convert::Infallible;
use crate::{components::PreTokenize, types::PreToken};

pub struct WhitespacePreTokenizer;
impl PreTokenize for WhitespacePreTokenizer {
    type Error = Infallible;
    fn pre_tokenize(&self, text: String) -> Result<Vec<PreToken>, Self::Error> {
        let mut result = vec![];
        let mut start = 0;
        for word in text.split_whitespace() {
            let idx = text[start..].find(word).unwrap();
            let abs_start = start + idx;
            let abs_end = abs_start + word.len();

            result.push(PreToken::new(word, (abs_start, abs_end)));
            start = abs_end;
        }
        
        Ok(result)
    }
}