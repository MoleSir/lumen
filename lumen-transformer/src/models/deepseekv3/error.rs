
#[thiserrorctx::context_error]
pub enum DeepSeekV3Error {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error(transparent)]
    Nn(#[from] lumen_nn::NnError),
}
