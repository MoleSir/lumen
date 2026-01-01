use std::sync::RwLock;

use crate::{FloatDType, IntTensor, Tensor, WithDType};
use super::{BinaryOp, Op, ReduceOp, UnaryOp};

pub trait AutogradMetaT<T: WithDType>: Default + Send + Sync {
    fn on_binary_op(lhs: &Tensor<T>, rhs: &Tensor<T>, op: BinaryOp) -> Self;
    fn on_binary_scalar_rhs_op(lhs: &Tensor<T>, rhs: T, op: BinaryOp) -> Self;
    fn on_binary_scalar_lhs_op(lhs: T, rhs: &Tensor<T>, op: BinaryOp) -> Self;
    fn on_unray_op(t: &Tensor<T>, op: UnaryOp<T>) -> Self; 
    fn on_pow_op(t: &Tensor<T>, e: T) -> Self;
    fn on_broadcast_op(t: &Tensor<T>) -> Self;
    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self;
    fn on_matmul_op(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Self;
    fn on_narrow_op(t: &Tensor<T>, dim: usize, start: usize, len: usize) -> Self;
    fn on_slice_op(t: &Tensor<T>, dim: usize, start: usize, end: usize, step: usize) -> Self;
    fn on_reshape_op(t: &Tensor<T>) -> Self;
    fn on_transpose_op(t: &Tensor<T>, dim1: usize, dim2: usize) -> Self;
    fn on_cat_op<A: AsRef<Tensor<T>>>(args: &[A], dim: usize) -> Self;
    fn on_permute_op(t: &Tensor<T>, dims: Vec<usize>) -> Self;
    fn on_copy_op(t: &Tensor<T>) -> Self;
    fn on_ifelse_op(mask: &Tensor<bool>, tv: Option<&Tensor<T>>, fv: Option<&Tensor<T>>) -> Self;
    fn on_index_select_op(t: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self;
    fn on_index_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self;
    fn on_scatter_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self;
    fn on_gather_op(src: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self;
}

pub struct AutogradInfo<T: FloatDType> {
    pub op: Option<Op<T>>,
    pub requires_grad: RwLock<bool>,
}

// pub enum AutogradInfo<T: FloatDType> {
//     Var(Option<Op<T>>),
//     Val
// }

impl<T: FloatDType> AutogradInfo<T>  {
    pub fn var() -> Self {
        Self {
            op: None,
            requires_grad: RwLock::new(true),
        }
    }

    pub fn val() -> Self {
        Self {
            op: None,
            requires_grad: RwLock::new(false),
        }
    }

    pub fn var_from_op(op: Op<T>) -> Self {
        Self {
            op: Some(op),
            requires_grad: RwLock::new(true),
        }
    }

    pub fn op(&self) -> Option<&Op<T>> {
        self.op.as_ref()
    }

    pub fn is_leaf(&self) -> bool {
        self.requires_grad() && self.op.is_none()
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad.read().unwrap().clone()
    }

    pub fn set_requires_grad(&self, mode: bool) {
        *self.requires_grad.write().unwrap() = mode;
    }
}

impl<T: FloatDType> Default for AutogradInfo<T> {
    fn default() -> Self {
        Self::val()
    }
} 

impl<T: FloatDType> AutogradMetaT<T> for AutogradInfo<T> {
    fn on_binary_op(lhs: &Tensor<T>, rhs: &Tensor<T>, op: BinaryOp) -> Self {
        if lhs.requires_grad() || rhs.requires_grad() {
            Self::var_from_op(Op::Binary(lhs.clone(), rhs.clone(), op))
        } else {
            Self::val()
        }
    }

    fn on_binary_scalar_rhs_op(lhs: &Tensor<T>, rhs: T, op: BinaryOp) -> Self {
        if lhs.requires_grad() {
            Self::var_from_op(Op::BinaryScalarRhs(lhs.clone(), rhs, op))
        } else {
            Self::val()
        }
    }

    fn on_binary_scalar_lhs_op(lhs: T, rhs: &Tensor<T>, op: BinaryOp) -> Self {
        if rhs.requires_grad() {
            Self::var_from_op(Op::BinaryScalarLhs(lhs, rhs.clone(), op))
        } else {
            Self::val()
        }
    }

    fn on_unray_op(t: &Tensor<T>, op: UnaryOp<T>) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Unary(t.clone(), op))
        } else {
            Self::val()
        }
    }

    fn on_pow_op(t: &Tensor<T>, e: T) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Pow(t.clone(), e))
        } else {
            Self::val()
        }
    }

    fn on_broadcast_op(t: &Tensor<T>) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Broadcast(t.clone()))
        } else {
            Self::val()
        }
    }

    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Reduce(t.clone(), op, dims.to_vec()))
        } else {
            Self::val()
        }
    }

    fn on_matmul_op(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Self {
        if lhs.requires_grad() || rhs.requires_grad() {
            Self::var_from_op(Op::Matmul(lhs.clone(), rhs.clone()))
        } else {
            Self::val()
        } 
    }

    fn on_narrow_op(t: &Tensor<T>, dim: usize, start: usize, len: usize) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Narrow(t.clone(), dim, start, len))
        } else {
            Self::val()
        } 
    }

    fn on_slice_op(t: &Tensor<T>, dim: usize, start: usize, end: usize, step: usize) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Slice(t.clone(), dim, start, end, step))
        } else {
            Self::val()
        } 
    }

    fn on_reshape_op(t: &Tensor<T>) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Reshape(t.clone()))
        } else {
            Self::val()
        } 
    }

    fn on_transpose_op(t: &Tensor<T>, dim1: usize, dim2: usize) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Transpose(t.clone(), dim1, dim2))
        } else {
            Self::val()
        } 
    }

    fn on_cat_op<A: AsRef<Tensor<T>>>(args: &[A], dim: usize) -> Self {
        if args.iter().any(|t| t.as_ref().requires_grad()) {
            let vec = args.iter().map(|a| a.as_ref().clone()).collect();
            Self::var_from_op(Op::Cat(vec, dim))
        } else {
            Self::val()
        }
    }

    fn on_permute_op(t: &Tensor<T>, dims: Vec<usize>) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Permute(t.clone(), dims))
        } else {
            Self::val()
        } 
    }

    fn on_copy_op(t: &Tensor<T>) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::Copy(t.clone()))
        } else {
            Self::val()
        }  
    }

    fn on_ifelse_op(mask: &Tensor<bool>, tv: Option<&Tensor<T>>, fv: Option<&Tensor<T>>) -> Self {
        match (tv, fv) {
            (Some(tv), Some(fv)) => {
                if tv.requires_grad() || fv.requires_grad() {
                    Self::var_from_op(Op::IfElse(mask.clone(), Some(tv.clone()), Some(fv.clone())))
                } else {
                    Self::val()
                }
            }
            (None, Some(fv)) => {
                if fv.requires_grad() {
                    Self::var_from_op(Op::IfElse(mask.clone(), None, Some(fv.clone())))
                } else {
                    Self::val()
                }
            }
            (Some(tv), None) => {
                if tv.requires_grad() {
                    Self::var_from_op(Op::IfElse(mask.clone(), Some(tv.clone()), None))
                } else {
                    Self::val()
                }
            }
            (None, None) => {
                Self::val()

            }
        }  
    }

    fn on_index_select_op(t: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::IndexSelect(t.clone(), indexes.clone(), dim))
        } else {
            Self::val()
        }  
    }

    fn on_index_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self {
        if init.requires_grad() || src.requires_grad() {
            Self::var_from_op(Op::IndexAdd(init.clone(), indexes.clone(), src.clone(), dim))
        } else {
            Self::val()
        }
    }

    fn on_scatter_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self {
        if init.requires_grad() || src.requires_grad() {
            Self::var_from_op(Op::IndexAdd(init.clone(), indexes.clone(), src.clone(), dim))
        } else {
            Self::val()
        }
    }

    fn on_gather_op(src: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self {
        if src.requires_grad() {
            Self::var_from_op(Op::Gather(src.clone(), indexes.clone(), dim))
        } else {
            Self::val()
        }
    }
}

#[derive(Default)]
pub struct NoAutograd;

#[allow(unused)]
impl<T: WithDType> AutogradMetaT<T> for NoAutograd {
    #[inline]
    fn on_binary_op(_: &Tensor<T>, _: &Tensor<T>, _: BinaryOp) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_binary_scalar_rhs_op(_: &Tensor<T>, _: T, _: BinaryOp) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_binary_scalar_lhs_op(lhs: T, rhs: &Tensor<T>, op: BinaryOp) -> Self {
        NoAutograd
    }
    
    #[inline]
    fn on_unray_op(_: &Tensor<T>, _: UnaryOp<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_broadcast_op(_: &Tensor<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_pow_op(t: &Tensor<T>, e: T) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_reduce_op(t: &Tensor<T>, _: &[usize], _: ReduceOp) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_matmul_op(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_narrow_op(t: &Tensor<T>, dim: usize, start: usize, len: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_slice_op(t: &Tensor<T>, dim: usize, start: usize, end: usize, step: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_reshape_op(t: &Tensor<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_transpose_op(t: &Tensor<T>, dim1: usize, dim2: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_cat_op<A: AsRef<Tensor<T>>>(args: &[A], dim: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_permute_op(t: &Tensor<T>, dims: Vec<usize>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_copy_op(t: &Tensor<T>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_ifelse_op(mask: &Tensor<bool>, tv: Option<&Tensor<T>>, fv: Option<&Tensor<T>>) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_index_select_op(t: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_index_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_scatter_add_op(init: &Tensor<T>, indexes: &IntTensor, src: &Tensor<T>, dim: usize) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_gather_op(src: &Tensor<T>, indexes: &IntTensor, dim: usize) -> Self {
        NoAutograd
    }
}