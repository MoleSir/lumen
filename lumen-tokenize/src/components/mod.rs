mod bpe;
use crate::Token;

pub trait Normalize {
    type Error: std::error::Error + 'static;
    fn normalize(&self, text: String) -> Result<String, Self::Error>;
}

pub trait PreTokenize {
    type Error: std::error::Error + 'static;
    fn pre_tokenize(&self, text: String) -> Result<Vec<String>, Self::Error>;
}

pub trait Model {
    type Error: std::error::Error + 'static;
    fn tokenize(&self, text: String) -> Result<Vec<Token>, Self::Error>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

pub trait PostProcess {
    type Error: std::error::Error + 'static;
    fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error>;
}

pub trait Decode {
    type Error: std::error::Error + 'static;
    fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error>;
}