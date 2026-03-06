mod error;
pub use error::*;
use crate::{components::{Decode, Model, Normalize, PostProcess, PreTokenize}, types::Token};
use crate::components::{NoNormalize, NoPreTokenize, NoPostProcess, SpaceDecode};

pub struct Tokenizer<
    M: Model, 
    N: Normalize = NoNormalize, 
    PT: PreTokenize = NoPreTokenize, 
    PP: PostProcess = NoPostProcess, 
    D: Decode = SpaceDecode
>  {
    pub normalizer: N,
    pub pre_tokenizer: PT,
    pub model: M,
    pub post_processor: PP,
    pub decoder: D,
}

impl<M: Model> Tokenizer<M> {
    pub fn new_model(model: M) -> Self {
        Self::new(NoNormalize, NoPreTokenize, model, NoPostProcess, SpaceDecode)
    }
}

impl<N, PT, M, PP, D> Tokenizer<M, N, PT, PP, D> 
where 
    N: Normalize,
    PT: PreTokenize,
    M: Model,
    PP: PostProcess,
    D: Decode,
{
    pub fn new(normalizer: N, pre_tokenizer: PT, model: M, post_processor: PP, decoder: D) -> Self {
        Self { normalizer, post_processor, pre_tokenizer, model, decoder } 
    }

    pub fn with_normalize<NewN: Normalize>(self, normalizer: NewN) -> Tokenizer<M, NewN, PT, PP, D> {
        Tokenizer::new(normalizer, self.pre_tokenizer, self.model, self.post_processor, self.decoder)
    }

    pub fn with_pre_tokenize<NewPT: PreTokenize>(self, pre_tokenizer: NewPT) -> Tokenizer<M, N, NewPT, PP, D> {
        Tokenizer::new(self.normalizer, pre_tokenizer, self.model, self.post_processor, self.decoder)
    }

    pub fn with_model<NewM: Model>(self, model: NewM) -> Tokenizer<NewM, N, PT, PP, D> {
        Tokenizer::new(self.normalizer, self.pre_tokenizer, model, self.post_processor, self.decoder)
    }

    pub fn with_post_process<NewPP: PostProcess>(self, post_processor: NewPP) -> Tokenizer<M, N, PT, NewPP, D> {
        Tokenizer::new(self.normalizer, self.pre_tokenizer, self.model, post_processor, self.decoder)
    }

    pub fn with_decode<NewD: Decode>(self, decoder: NewD) -> Tokenizer<M, N, PT, PP, NewD> {
        Tokenizer::new(self.normalizer, self.pre_tokenizer, self.model, self.post_processor, decoder)
    }

    pub fn encode(&self, text: &str) -> TokenizeResult<Vec<Token>> {
        let normalized = self.normalizer
            .normalize(text)
            .map_err(|e| TokenizeError::Normalize(Box::new(e)))?;

        let pre_tokens = self.pre_tokenizer
            .pre_tokenize(normalized)
            .map_err(|e| TokenizeError::PreTokenize(Box::new(e)))?;

        let tokens = self.model
            .tokenize(pre_tokens)    
            .map_err(|e| TokenizeError::Model(Box::new(e)))?;

        let tokens = self.post_processor
            .process(tokens)
            .map_err(|e| TokenizeError::PostProcess(Box::new(e)))?;

        Ok(tokens)
    }

    pub fn decode(&self, ids: &[u32]) -> TokenizeResult<String> {
        let tokens = ids.iter()
            .filter(|id| !self.post_processor.is_special_token_id(**id))
            .filter_map(|&id| self.model.id_to_token(id))
            .collect::<Vec<_>>();
        let s = self.decoder
            .decode(tokens)
            .map_err(|e| TokenizeError::Decode(Box::new(e)))?;
        Ok(s)
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, convert::Infallible};
    use crate::{components::{Model, Normalize, PostProcess, PreTokenize, SpaceDecode}, types::{PreToken, Token}};
    use super::Tokenizer;

    struct LowercaseNormalizer;
    impl Normalize for LowercaseNormalizer {
        type Error = Infallible;
        fn normalize(&self, text: &str) -> Result<String, Infallible> {
            Ok(text.to_lowercase())
        }
    }

    struct WhitespacePreTokenizer;
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

    struct SimpleVocabModel {
        vocab: HashMap<String, u32>,
        ids_to_tokens: HashMap<u32, String>,
    }

    impl Model for SimpleVocabModel {
        type Error = Infallible;
        fn tokenize(&self, pre_tokens: Vec<PreToken>) -> Result<Vec<Token>, Self::Error> {
            let tokens = pre_tokens.into_iter()
                .map(|pt| {
                    let id = *self.vocab.get(&pt.value).unwrap_or(&0);
                    Token::new(id, pt.value, pt.offset)
                })
                .collect();

            Ok(tokens)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.ids_to_tokens.get(&id).cloned()
        }
    }

    struct BertPostProcessor;
    impl PostProcess for BertPostProcessor {
        type Error = Infallible;

        fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error> {
            let mut res = vec![Token::new(101, "[CLS]", (0, 0))];
            res.extend(tokens);
            res.push(Token::new(102, "[SEP]", (0, 0)));
            Ok(res)
        }

        fn is_special_token_id(&self, id: u32) -> bool {
            if id == 101 || id == 102 {
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn test_fake_encode() {
        let mut vocab = HashMap::new();
        vocab.insert("hello".into(), 1);
        vocab.insert("world".into(), 2);
        let mut rev_vocab = HashMap::new();
        rev_vocab.insert(1, "hello".into());
        rev_vocab.insert(2, "world".into());

        let tokenizer = Tokenizer::new(
            LowercaseNormalizer, 
            WhitespacePreTokenizer, 
            SimpleVocabModel { vocab, ids_to_tokens: rev_vocab },
            BertPostProcessor,
            SpaceDecode,
        );

        let input = "Hello World";

        let output = tokenizer.encode(input).unwrap();
        assert_eq!(
            output, 
            vec![
                Token::new(101, "[CLS]", (0, 0)), 
                Token::new(1, "hello", (0, 5)), 
                Token::new(2, "world", (6, 11)), 
                Token::new(102, "[SEP]", (0, 0))]);

        let ids: Vec<u32> = output.iter().map(|t| t.id).collect();
        println!("{:?}", ids);
        let input_rec = tokenizer.decode(&ids).unwrap();
        assert_eq!(input_rec, input.to_lowercase());
        
    }
}