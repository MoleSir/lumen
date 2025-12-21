use crate::{FloatDType, Tensor, WithDType};
use super::{BinaryOp, Op, ReduceOp, UnaryOp};

pub trait AutogradMetaT<T: WithDType>: Default + Send + Sync {
    fn on_binary_op(lhs: &Tensor<T>, rhs: &Tensor<T>, op: BinaryOp) -> Self;
    fn on_binary_scalar_op(lhs: &Tensor<T>, rhs: T, op: BinaryOp) -> Self;
    fn on_unray_op(t: &Tensor<T>, op: UnaryOp) -> Self; 
    fn on_pow_op(t: &Tensor<T>, e: T) -> Self;
    fn on_broadcast_op(t: &Tensor<T>) -> Self;
    fn on_reduce_op(t: &Tensor<T>, dims: &[usize], op: ReduceOp) -> Self;
    fn on_reduce_all_op(t: &Tensor<T>, op: ReduceOp) -> Self;
    fn on_matmul_op(lhs: &Tensor<T>, rhs: &Tensor<T>) -> Self;
    fn on_narrow_op(t: &Tensor<T>, dim: usize, start: usize, len: usize) -> Self;
    fn on_slice_op(t: &Tensor<T>, dim: usize, start: usize, end: usize, step: usize) -> Self;
    fn on_reshape_op(t: &Tensor<T>) -> Self;
    fn on_transpose_op(t: &Tensor<T>, dim1: usize, dim2: usize) -> Self;
    fn on_cat_op<A: AsRef<Tensor<T>>>(args: &[A], dim: usize) -> Self;
    fn on_permute_op(t: &Tensor<T>, dims: Vec<usize>) -> Self;
    fn on_copy_op(t: &Tensor<T>) -> Self;
    fn on_ifelse_op(mask: &Tensor<bool>, tv: Option<&Tensor<T>>, fv: Option<&Tensor<T>>) -> Self;
}

// pub struct AutogradInfo<T: FloatDType> {
//     pub op: Option<Op<T>>,
//     pub requires_grad: bool,
// }

pub enum AutogradInfo<T: FloatDType> {
    Var(Option<Op<T>>),
    Val
}

impl<T: FloatDType> AutogradInfo<T>  {
    pub fn var() -> Self {
        Self::Var(None)
    }

    pub fn val() -> Self {
        Self::Val
    }

    pub fn var_from_op(op: Op<T>) -> Self {
        Self::Var(Some(op))
    }

    pub fn op(&self) -> Option<&Op<T>> {
        match self {
            Self::Val => None,
            Self::Var(op) => op.as_ref()
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Val => false,
            Self::Var(op) => op.is_none(),
        }
    }

    pub fn requires_grad(&self) -> bool {
        match self {
            Self::Val => false,
            Self::Var(_) => true,
        }
    }
}

impl<T: FloatDType> Default for AutogradInfo<T> {
    fn default() -> Self {
        Self::Val
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

    fn on_binary_scalar_op(lhs: &Tensor<T>, rhs: T, op: BinaryOp) -> Self {
        if lhs.requires_grad() {
            Self::var_from_op(Op::BinaryScalar(lhs.clone(), rhs, op))
        } else {
            Self::val()
        }
    }

    fn on_unray_op(t: &Tensor<T>, op: UnaryOp) -> Self {
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

    fn on_reduce_all_op(t: &Tensor<T>, op: ReduceOp) -> Self {
        if t.requires_grad() {
            Self::var_from_op(Op::ReduceAll(t.clone(), op))
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
    fn on_binary_scalar_op(_: &Tensor<T>, _: T, _: BinaryOp) -> Self {
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
    fn on_pow_op(t: &Tensor<T>, e: T) -> Self {
        NoAutograd
    }

    #[inline]
    fn on_reduce_op(t: &Tensor<T>, _: &[usize], _: ReduceOp) -> Self {
        NoAutograd
    }
    
    #[inline]
    fn on_reduce_all_op(t: &Tensor<T>, op: ReduceOp) -> Self {
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
}