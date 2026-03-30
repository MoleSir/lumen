
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("normalize error: {0}")]
    Normalize(Box<dyn std::error::Error>),

    #[error("pre tokenize error: {0}")]
    PreTokenize(Box<dyn std::error::Error>),

    #[error("model error: {0}")]
    Model(Box<dyn std::error::Error>),

    #[error("post process error: {0}")]
    PostProcess(Box<dyn std::error::Error>),

    #[error("decode error: {0}")]
    Decode(Box<dyn std::error::Error>),
    
}

pub type TokenizeResult<T> = std::result::Result<T, TokenizeError>;