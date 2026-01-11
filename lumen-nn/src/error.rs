use lumen_core::Shape;


#[thiserrorctx::context_error]
pub enum NnError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error("can't found param {0} in {1}")]
    ParamNotFound(String, &'static str),

    #[error("shape unmatch when load param: expect {0}, but got {1}")]
    ShapeUnmatchWhenLoadParam(Shape, Shape),
}