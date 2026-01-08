
#[thiserrorctx::context_error]
pub enum Gpt2Error {
    #[error(transparent)]
    Core(#[from] lumen_core::CtxError),
}
