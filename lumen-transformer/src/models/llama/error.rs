
#[thiserrorctx::context_error]
pub enum LlamaError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] std::array::TryFromSliceError),

    #[error("invalid format {0}")]
    InvalidFormat(String),

    #[error("Data offset out of range, total {0}, but try get {1}")]
    DataOffsetOutOfRange(usize, usize),
}

