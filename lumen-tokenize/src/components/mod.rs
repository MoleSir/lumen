use std::convert::Infallible;

use crate::types::{PreToken, Token};

pub mod normalize;
pub mod pre_tokenize;
pub mod model;
pub mod post_process;
pub mod decode;

/// Takes care of pre-processing strings.
pub trait Normalize {
    type Error: std::error::Error + 'static;
    fn normalize(&self, text: &str) -> Result<String, Self::Error>;
}

/// The `PreTokenize` is in charge of doing the pre-segmentation step. It splits the given string in multiple substrings.
pub trait PreTokenize {
    type Error: std::error::Error + 'static;
    fn pre_tokenize(&self, text: String) -> Result<Vec<PreToken>, Self::Error>;
}

/// Represents a model used during Tokenization (like BPE or Word or Unigram).
pub trait Model {
    type Error: std::error::Error + 'static;
    fn tokenize(&self, pre_tokens: Vec<PreToken>) -> Result<Vec<Token>, Self::Error>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

/// A `PostProcess` has the responsibility to post process an encoded output of the `Tokenizer`.
/// It adds any special tokens that a language model would require.
pub trait PostProcess {
    type Error: std::error::Error + 'static;
    fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error>;
    fn is_special_token_id(&self, id: u32) -> bool;
}

pub trait Decode {
    type Error: std::error::Error + 'static;
    fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error>;
}

pub struct NoNormalize;
impl Normalize for NoNormalize {
    type Error = Infallible;
    fn normalize(&self, text: &str) -> Result<String, Self::Error> {
        Ok(text.to_string())
    }
}

pub struct NoPostProcess;
impl PostProcess for NoPostProcess {
    type Error = Infallible;

    #[inline]
    fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error> {
        Ok(tokens)
    }

    #[inline]
    fn is_special_token_id(&self, _id: u32) -> bool {
        false
    }
}

pub struct NoPreTokenize;
impl PreTokenize for NoPreTokenize {
    type Error = Infallible;
    fn pre_tokenize(&self, text: String) -> Result<Vec<crate::types::PreToken>, Self::Error> {
        let text_len = text.len();
        Ok(vec![PreToken { value: text, offset: (0, text_len) }])
    }
}

pub struct SpaceDecode;
impl Decode for SpaceDecode {
    type Error = Infallible;
    fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error> {
        Ok(tokens.join(" "))
    }
} 