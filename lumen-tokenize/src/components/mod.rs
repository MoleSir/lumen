use crate::token::Token;

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
}

pub trait PostProcessor {
    type Error: std::error::Error + 'static;
    fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error>;
}

// /// A `PostProcess` has the responsibility to post process an encoded output of the `Tokenizer`.
// /// It adds any special tokens that a language model would require.
// pub trait PostProcess {
//     type Error: std::error::Error + 'static;
//     fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error>;
//     fn is_special_token_id(&self, id: u32) -> bool;
// }

// pub trait Decode {
//     type Error: std::error::Error + 'static;
//     fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error>;
// }

// pub struct NoNormalize;
// impl Normalize for NoNormalize {
//     type Error = Infallible;
//     fn normalize(&self, text: &str) -> Result<String, Self::Error> {
//         Ok(text.to_string())
//     }
// }

// pub struct NoPostProcess;
// impl PostProcess for NoPostProcess {
//     type Error = Infallible;

//     #[inline]
//     fn process(&self, tokens: Vec<Token>) -> Result<Vec<Token>, Self::Error> {
//         Ok(tokens)
//     }

//     #[inline]
//     fn is_special_token_id(&self, _id: u32) -> bool {
//         false
//     }
// }

// pub struct NoPreTokenize;
// impl PreTokenize for NoPreTokenize {
//     type Error = Infallible;
//     fn pre_tokenize(&self, text: String) -> Result<Vec<crate::types::PreToken>, Self::Error> {
//         let text_len = text.len();
//         Ok(vec![PreToken { value: text, offset: (0, text_len) }])
//     }
// }

// pub struct SpaceDecode;
// impl Decode for SpaceDecode {
//     type Error = Infallible;
//     fn decode(&self, tokens: Vec<String>) -> Result<String, Self::Error> {
//         Ok(tokens.join(" "))
//     }
// } 