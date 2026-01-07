use std::str::Utf8Error;
use crate::{DType, Slice, Shape};

#[thiserrorctx::Error]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error("Index '{index}' out of range at storage({storage_len}) in take method")]
    IndexOutOfRangeTake {
        storage_len: usize,
        index: usize,
    },

    #[error("index '{index}' out of range range({max_size}) in {op}")]
    IndexOutOfRange {
        max_size: usize,
        index: usize,
        op: &'static str,
    },

    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex {
        shape: Shape,
        dims: Vec<usize>,
        op: &'static str,
    },

    #[error("try to repeat {repeats} for shape {shape}")]
    RepeatRankOutOfRange {
        repeats: Shape,
        shape: Shape,
    },

    // === Shape Errors ===
    #[error("unexpected element size in {op}, expected: {expected}, got: {got}")]
    ElementSizeMismatch {
        expected: usize,
        got: usize,
        op: &'static str
    },

    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error("requires contiguous {op}")]
    RequiresContiguous { op: &'static str },

    #[error("invalid index in {op}")]
    InvalidIndex {
        index: usize,
        size: usize,
        op: &'static str 
    },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat {
        dim: usize,
        first_shape: Shape,
        n: usize,
        nth_shape: Shape,
    },

    #[error("source Tensor shape {src:?} mismatch with condition shape {condition:?}")]
    ShapeMismatchMaskedSelect {
        src: Shape,
        condition: Shape, 
    },

    #[error("mask Tensor shape {mask:?} mismatch with {who} shape")]
    ShapeMismatchSelect {
        mask: Shape,
        who: &'static str,
    },

    #[error("dst Tensor shape {dst:?} mismatch with src Tensor {src} shape")]
    ShapeMismatchCopyFrom {
        dst: Shape,
        src: Shape,
    },

    // === Op Specific Errors ===
    #[error("slice invalid args {msg}: {shape:?}, dim: {dim}, slice: {slice}")]
    SliceInvalidArgs {
        shape: Shape,
        dim: usize,
        slice: Slice,
        msg: &'static str,
    },

    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error("can squeeze {dim} dim of {shape:?}(not 1)")]
    SqueezeDimNot1 {
        shape: Shape,
        dim: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("rand error because {0}")]
    Rand(String),

    #[error("Tensor is not a scalar")]
    NotScalar,

    // === View ===
    #[error("len mismatch with lhs {lhs} and rhs {rhs} in {op}")]
    LenMismatchVector {
        lhs: usize,
        rhs: usize,
        op: &'static str,
    },

    #[error("shape mismatch with lhs {lhs:?} and rhs {rhs:?} in {op}")]
    ShapeMismatchMatrix {
        lhs: (usize, usize),
        rhs: (usize, usize),
        op: &'static str,
    },

    #[error("index {index} of out range in {len} len vector")]
    VectorIndexOutOfRange {
        len: usize,
        index: usize,
    },

    #[error("{position} index {index} of out range in {len} len matrix")]
    MatrixIndexOutOfRange {
        len: usize,
        index: usize,
        position: &'static str,
    },

    #[error("backward not support '{0}'")]
    BackwardNotSupported(&'static str),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] Utf8Error),

    // === Utils ===
    #[error("{context}\n{inner}")]
    Context {
        inner: Box<Self>,
        context: String,
    },

    /// User generated error message
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()))?
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()))?
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()))?
    };
}