#[thiserrorctx::context_error]
pub enum Qwen2Error {
    #[error(transparent)]
    SafeTensors(#[from] lumen_io::safetensors::SafeTensorsCtxError),
    
    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Nn(#[from] lumen_nn::NnCtxError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] std::array::TryFromSliceError),

    #[error("invalid format {0}")]
    InvalidFormat(String),

    #[error("Data offset out of range, total {0}, but try get {1}")]
    DataOffsetOutOfRange(usize, usize),
}

