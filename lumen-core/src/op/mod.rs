use crate::{FloatDType, Tensor};
mod gradmeta;
mod gradstore;
pub use gradmeta::*;
pub use gradstore::*;
mod backprop;
mod test;

#[derive(Clone)]
pub enum Op<T: FloatDType> {
    Binary(Tensor<T>, Tensor<T>, BinaryOp),
    BinaryScalar(Tensor<T>, T, BinaryOp),
    Unary(Tensor<T>, UnaryOp),
    Reduce(Tensor<T>, ReduceOp, Vec<usize>),
    ReduceAll(Tensor<T>, ReduceOp),
    Matmul(Tensor<T>, Tensor<T>),
    Broadcast(Tensor<T>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Exp,
    Ln,

    Sin,
    Cos,
    Tanh,

    Abs,
    Neg,
    Sqr,
    Sqrt,

    Recip,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    Silu,

    Floor,
    Ceil,
    Round,
    Sign,
}
