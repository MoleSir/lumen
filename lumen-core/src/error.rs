use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("data length must match shape dimensions")]
    DataLenShapeUnmatch,

    #[error("Shape unmath in reshape")]
    ReshapeShapeUnmatch,

    #[error("dimensions unmatch")]
    DimensionsUnmatch,

    #[error("indices and shape unmatch")]
    IndicesShapeUnmatch,

    #[error("index out of range")]
    IndexOutOfRange,

    #[error("different shape")]
    DifferentShape,

    #[error("apply `view` in uncontiguous tensor")]
    ViewUnContiguous,

    #[error("apply an zero size slice in tensor")]
    SliceZeroSize,

    #[error("shape error when mat mul")]
    MatMulShapeError,
} 