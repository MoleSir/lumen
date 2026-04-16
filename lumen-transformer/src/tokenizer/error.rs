#[thiserrorctx::context_error]
pub enum TokenzieError {
    #[error(transparent)]
    Serde(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]   
    Tokenizers(#[from] tokenizers::Error),
}
