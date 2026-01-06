
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),
}

pub type LlamaResult<T> = Result<T, LlamaError>;