use lumen_core::Shape;

#[thiserrorctx::context_error]
pub enum NnError {
    #[error(transparent)]
    Core(#[from] lumen_core::Error),

    #[error("can't found param {0} in {1}")]
    ParamNotFound(String, &'static str),

    #[error("shape unmatch when load param: expect {0}, but got {1}")]
    ShapeUnmatchWhenLoadParam(Shape, Shape),

    #[error(transparent)]
    SafeTensors(#[from] lumen_io::safetensors::SafeTensorsCtxError),

    #[error("head_size {0} can't divde by num_head {1}")]
    HeadSizeCannotDivideByNumhead(usize, usize),

    #[error("head_size {0} can't divde by kv_num_head {1}")]
    HeadSizeCannotDivideByKvNumhead(usize, usize),

    #[error("unsupport shape {0} of input in batch norm 1d")]
    BatchNorm1dUnsupportShape(Shape),

    #[error("drop_p {0} invalid(not in [0, 1)])")]
    DropoutInvalid(f64)
}
