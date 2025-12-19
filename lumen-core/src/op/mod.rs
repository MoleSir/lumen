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
    BinaryScalar(Tensor<T>, T, BinaryScalarOp),
    Unary(Tensor<T>, UnaryOp),
    Reduce(Tensor<T>, ReduceOp, Vec<usize>),
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
pub enum BinaryScalarOp {
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
    Sin,
    Cos,
    Abs,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    Silu,
    Tanh,
    Floor,
    Ceil,
    Round,
    Sign,
    Ln,
}
