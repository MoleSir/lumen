
#[thiserrorctx::context_error]
pub enum LlamaError {
    #[error(transparent)]
    Core(#[from] lumen_core::CtxError),
}
