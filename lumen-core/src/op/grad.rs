use std::sync::RwLock;
use crate::{FloatDType, Tensor, WithDType};
use super::{BinaryOp, Op, ReduceOp, UnaryOp};

pub trait AutogradMetaT<T: WithDType>: Default + Send + Sync {
    fn on_binary_op(lhs: &Tensor<T>, rhs: &Tensor<T>, op: BinaryOp) -> Self;
    fn on_unray_op(t: &Tensor<T>, op: UnaryOp) -> Self; 
    fn on_broadcast_op(t: &Tensor<T>) -> Self;
    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self;
}

pub struct AutogradInfo<T: FloatDType> {
    pub grad: RwLock<Option<Tensor<T>>>,
    pub op: Option<Op<T>>,
    pub requires_grad: bool,
}

impl<T: FloatDType> AutogradInfo<T>  {
    pub fn var() -> Self {
        Self {
            grad: RwLock::new(None),
            op: None,
            requires_grad: true
        }
    }

    pub fn none() -> Self {
        Default::default()
    }

    pub fn var_with_op(op: Op<T>) -> Self {
        Self {
            grad: RwLock::new(None),
            op: Some(op),
            requires_grad: true
        }
    }
}

impl<T: FloatDType> Default for AutogradInfo<T> {
    fn default() -> Self {
        Self {
            grad: RwLock::new(None),
            op: None,
            requires_grad: false,
        }
    }
} 

impl<T: FloatDType> AutogradMetaT<T> for AutogradInfo<T> {
    fn on_binary_op(lhs: &Tensor<T>, rhs: &Tensor<T>, op: BinaryOp) -> Self {
        if lhs.requires_grad() || rhs.requires_grad() {
            Self::var_with_op(Op::Binary(lhs.clone(), rhs.clone(), op))
        } else {
            Self::none()
        }
    }

    fn on_unray_op(t: &Tensor<T>, op: UnaryOp) -> Self {
        if t.requires_grad() {
            Self::var_with_op(Op::Unary(t.clone(), op))
        } else {
            Self::none()
        }
    }

    fn on_broadcast_op(t: &Tensor<T>) -> Self {
        if t.requires_grad() {
            Self::var_with_op(Op::Broadcast(t.clone()))
        } else {
            Self::none()
        }
    }

    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self {
        if t.requires_grad() {
            Self::var_with_op(Op::Reduce(t.clone(), op, dims.to_vec()))
        } else {
            Self::none()
        }
    }
}

#[derive(Default)]
pub struct NoAutograd;

impl<T: WithDType> AutogradMetaT<T> for NoAutograd {
    #[inline]
    fn on_binary_op(_: &Tensor<T>, _: &Tensor<T>, _: BinaryOp) -> Self {
        NoAutograd
    }
    
    #[inline]
    fn on_unray_op(_: &Tensor<T>, _: UnaryOp) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_broadcast_op(_: &Tensor<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self {
        NoAutograd
    }
}