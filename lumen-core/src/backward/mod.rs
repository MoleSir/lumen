mod imple;

use crate::{Range, Tensor};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Backward {
    Root,
    Unary(UnaryTensorBackward),
    Binary(BinaryTensorBackward),
    Multi(MultiTensorBackward),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum UnaryTensorBackward {
    Exp(Tensor),
    Pow(Tensor, f64),
    Log(Tensor, f64),
    Tanh(Tensor),
    Sigmoid(Tensor),
    ReLU(Tensor),
    Sin(Tensor),
    Cos(Tensor),
    Abs(Tensor),
    View(Tensor),
    Slice(Tensor, Vec<Range>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BinaryTensorBackward {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    MatMul(Tensor, Tensor),
    BroadcastAdd(Tensor, Tensor),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum MultiTensorBackward {
    Stack(Vec<Tensor>)
}

impl UnaryTensorBackward {
    pub(crate) fn backward(&self, tensor: &Tensor) {
        match self {
            Self::Exp(x) => imple::exp_backward(tensor, x),
            Self::Pow(x, e) => imple::pow_backward(tensor, x, *e),
            Self::Log(x, base) => imple::log_backward(tensor, x, *base),
            Self::Tanh(x) => imple::tanh_backward(tensor, x),
            Self::Sigmoid(x) => imple::sigmoid_backward(tensor, x),
            Self::ReLU(x) => imple::relu_backward(tensor, x),
            Self::Sin(x) => imple::sin_backward(tensor, x),
            Self::Cos(x) => imple::cos_backward(tensor, x),
            Self::Abs(x) => imple::abs_backward(tensor, x),
            Self::View(pre) => imple::view_backward(tensor, pre),
            Self::Slice(pre, ranges) => imple::slice_backward(tensor, pre, &ranges),
        }
    }

    pub(crate) fn operand(&self) -> &Tensor {
        match self {
            Self::Exp(pre) => pre,
            Self::Pow(base, _) => base,
            Self::Log(e, _) => e,
            Self::Tanh(pre) => pre,
            Self::Sigmoid(pre) => pre,
            Self::ReLU(pre) => pre,
            Self::Sin(pre) => pre,
            Self::Cos(pre) => pre,
            Self::Abs(pre) => pre,
            Self::View(t) => t,
            Self::Slice(t, _) => t,
        }
    }
}

impl BinaryTensorBackward {
    pub(crate) fn backward(&self, tensor: &Tensor) {
        match self {
            Self::Add(lhs, rhs) => imple::add_backward(tensor, lhs, rhs),
            Self::BroadcastAdd(lhs, rhs) => imple::broadcast_add_backward(tensor, lhs, rhs),
            Self::Sub(lhs, rhs) => imple::sub_backward(tensor, lhs, rhs),
            Self::Mul(lhs, rhs) => imple::mul_backward(tensor, lhs, rhs),
            Self::Div(lhs, rhs) => imple::div_backward(tensor, lhs, rhs),
            Self::MatMul(lhs, rhs) => imple::matmul_backward(tensor, lhs, rhs),
        }
    }

    pub(crate) fn operands(&self) -> (&Tensor, &Tensor) {
        match self {
            Self::Add(l, r) => (l, r),
            Self::BroadcastAdd(l, r) => (l, r),
            Self::Sub(l, r) => (l, r),
            Self::Mul(l, r) => (l, r),
            Self::Div(l, r) => (l, r),
            Self::MatMul(l, r) => (l, r),
        }
    }
}

impl MultiTensorBackward {
    pub(crate) fn backward(&self, tensor: &Tensor) {
        match self {
            Self::Stack(tensors) => imple::stack_backward(tensor, &tensors),
        }
    }

    pub(crate) fn operands(&self) -> &[Tensor] {
        match self {
            Self::Stack(tensors) => tensors
        }
    }
}