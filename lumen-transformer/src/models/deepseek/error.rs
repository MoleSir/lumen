
#[thiserrorctx::context_error]
pub enum DeepSeekError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),
}
