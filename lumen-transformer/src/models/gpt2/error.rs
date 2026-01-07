
#[thiserrorctx::Error]
pub enum Gpt2Error {
    #[error(transparent)]
    Core(#[from] lumen_core::ErrorCtx),
}
