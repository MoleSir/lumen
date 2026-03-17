
#[thiserrorctx::context_error]
pub enum Gpt2Error {
    #[error(transparent)]
    SafeTensors(#[from] lumen_io::safetensors::SafeTensorsCtxError),
    
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Nn(#[from] lumen_nn::NnCtxError),
}

