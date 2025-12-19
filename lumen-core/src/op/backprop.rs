use std::collections::HashMap;
use crate::{BinaryOp, FloatDType, Op, ReduceOp, Tensor, TensorId, UnaryOp};
use super::GradStore;

impl<T: FloatDType> Tensor<T> {

    pub fn backward(&self) -> crate::Result<GradStore<T>> {
        let sorted_nodes = self.sorted_nodes();
        let mut grads = GradStore::new();
        grads.insert(self, self.ones_like()?);

        for node in sorted_nodes.iter() {
            match node.op() {
                None => {
                    assert!(node.is_leaf());
                    continue
                }
                Some(op) => {
                    let grad = grads
                        .remove(node)
                        .expect("grad not populated");

                    match op {
                        //=========================================================================================//
                        //           Binary
                        //=========================================================================================//
                        Op::Binary(lhs, rhs, BinaryOp::Add) => {
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            *rhs_sum_grad = rhs_sum_grad.add(&grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Sub) => {
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            *rhs_sum_grad = rhs_sum_grad.sub(&grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                            let lhs_grad = grad.mul(rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                            let rhs_grad = grad.mul(lhs)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Div) => {
                            let lhs_grad = grad.div(rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                            let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr())?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            *rhs_sum_grad = rhs_sum_grad.sub(&rhs_grad)?;
                        }
                        
                        //=========================================================================================//
                        //           BinaryScalar
                        //=========================================================================================//
                        Op::BinaryScalar(lhs, _, BinaryOp::Add) => {
                            // y = x + c => dy/dx = 1
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        }
                        Op::BinaryScalar(lhs, _, BinaryOp::Sub) => {
                            // y = x - c => dy/dx = 1
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&grad)?;
                        }
                        Op::BinaryScalar(lhs, rhs, BinaryOp::Mul) => {
                            // y = x * c => dy/dx = c
                            let lhs_grad = grad.mul_scalar(*rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        }
                        Op::BinaryScalar(lhs, rhs, BinaryOp::Div) => {
                            // y = x / c => dy/dx = 1/c
                            let lhs_grad = grad.div_scalar(*rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
                        }

                        //=========================================================================================//
                        //           Unary
                        //=========================================================================================//
                        Op::Unary(_, UnaryOp::Ceil) => Err(crate::Error::BackwardNotSupported("ceil"))?,
                        Op::Unary(_, UnaryOp::Floor) => Err(crate::Error::BackwardNotSupported("floor"))?,
                        Op::Unary(_, UnaryOp::Round) => Err(crate::Error::BackwardNotSupported("round"))?,                        
                        Op::Unary(_, UnaryOp::Sign) => Err(crate::Error::BackwardNotSupported("sign"))?,
                        Op::Unary(arg, UnaryOp::Exp) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&(&grad * *node))?
                        }
                        Op::Unary(arg, UnaryOp::Ln) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&(grad / arg))?
                        }
                        Op::Unary(arg, UnaryOp::Sin) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&(&grad * arg.cos()))?
                        }
                        Op::Unary(arg, UnaryOp::Cos) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.sub(&(&grad * arg.sin()))?
                        }
                        Op::Unary(arg, UnaryOp::Tanh) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let minus_dtanh = node.sqr() - T::one();
                            *sum_grad = sum_grad.sub(&(&grad * &minus_dtanh))?
                        }
                        Op::Unary(arg, UnaryOp::Sqr) => {
                            let arg_grad = arg.mul(&grad)?.affine(T::two(), T::zero())?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad)?
                        }
                        Op::Unary(arg, UnaryOp::Sqrt) => {
                            let arg_grad = grad.div(*node)?.affine(T::half(), T::zero())?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad)?
                        }
                        Op::Unary(arg, UnaryOp::Abs) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let ones = arg.ones_like()?;
                            let abs_grad = arg.ge(&arg.zeros_like()?)?.select(&ones, &ones.neg())?;
                            *sum_grad = sum_grad.add(&(&grad * abs_grad))?
                        }
                        Op::Unary(arg, UnaryOp::Neg) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.sub(&grad)?
                        }
                        Op::Unary(arg, UnaryOp::Recip) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let grad = grad / arg.sqr();
                            *sum_grad = sum_grad.sub(&grad)?
                        }
                        Op::Unary(arg, UnaryOp::Gelu) => {
                            // let sum_grad = grads.or_insert(arg)?;
                            // let cube = arg.powf(3.)?;
                            // let tanh = (0.0356774 * &cube + (0.797885 * arg)?)?.tanh()?;
                            // let gelu_grad = (((0.5 * &tanh)?
                            //     + (0.0535161 * cube + (0.398942 * arg)?)? * (1. - tanh.powf(2.)?))?
                            //     + 0.5)?;
                            // *sum_grad = sum_grad.add(&(&grad * gelu_grad)?)?
                        }
                        Op::Unary(arg, UnaryOp::Erf) => {
                            let sum_grad = grads.or_insert(arg)?;
                            // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
                            let erf_grad = arg.sqr().neg().exp() * (T::two() / T::pi().sqrt());
                            *sum_grad = sum_grad.add(&(&grad * erf_grad))?
                        }
                        Op::Unary(arg, UnaryOp::GeluErf) => {
                            let sum_grad = grads.or_insert(arg)?;
                            // d/dx gelu_erf(x) = 0.5 + 0.398942 e^(-x^2/2) x + 0.5 erf(x/sqrt(2))
                            let neg_half_square = arg.sqr().neg() / T::two();
                            let scaled_exp_arg = neg_half_square.exp() * arg * T::from_f64(0.398942);
                            let arg_scaled_sqrt = arg / T::two().sqrt();
                            let erf_scaled_sqrt = arg_scaled_sqrt.erf() / T::two();
                            let gelu_erf_grad = scaled_exp_arg + erf_scaled_sqrt + T::half();
                            *sum_grad = sum_grad.add(&(&grad * gelu_erf_grad))?;
                        }
                        Op::Unary(arg, UnaryOp::Relu) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let relu_grad = arg.ge(&arg.zeros_like()?)?.to_dtype::<T>();
                            *sum_grad = sum_grad.add(&(&grad * relu_grad))?
                        }
                        Op::Unary(arg, UnaryOp::Silu) => {
                            let sum_grad = grads.or_insert(arg)?;
                            // d/dx silu = sigmoid(x) * (1 + x * (1 - sigmoid(x))) = sigmoid(x) * (1 - node) + node
                            let sigmoid_arg = (arg.neg().exp() + T::one()).recip();
                            // TODO: 1 - Tensor
                            let silu_grad = &sigmoid_arg * (node.neg() + T::one()) + *node;
                            *sum_grad = sum_grad.add(&(&grad * silu_grad))?
                        }

                        //=========================================================================================//
                        //           Matmul
                        //=========================================================================================//
                        Op::Matmul(lhs, rhs) => {    
                            let lhs_grad = grad.matmul(&rhs.transpose_last()?)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad)?;
    
                            let rhs_grad = lhs.transpose_last()?.matmul(&grad)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad)?;
                        }

                        //=========================================================================================//
                        //           Reduce
                        //=========================================================================================//
                        Op::Reduce(arg, ReduceOp::Sum, reduced_dims) => {
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&grad)?;
                        }
                        Op::Reduce(arg, ReduceOp::Max, reduced_dims) => {
                            let node = Self::broadcast_back(arg, node, reduced_dims)?;
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let grad = node.eq(arg)?.to_dtype().mul(&grad)?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                        }
                        Op::Reduce(arg, ReduceOp::Min, reduced_dims) => {
                            let node = Self::broadcast_back(arg, node, reduced_dims)?;
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let grad = node.eq(arg)?.to_dtype().mul(&grad)?;
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&grad.broadcast_as(sum_grad.dims())?)?;
                        }

                        //=========================================================================================//
                        //           Broadcast
                        //=========================================================================================//
                        Op::Broadcast(arg) => {
                            let arg_dims = arg.dims();
                            let node_dims = node.dims();
                            let left_dims = node_dims.len() - arg_dims.len();
                            let mut sum_dims: Vec<usize> = (0..left_dims).collect();
                            for (dim, (node_dim, arg_dim)) in node_dims[left_dims..]
                                .iter()
                                .zip(arg_dims.iter())
                                .enumerate()
                            {
                                if node_dim != arg_dim {
                                    sum_dims.push(dim + left_dims)
                                }
                            }
    
                            let mut arg_grad = grad;
                            for &dim in sum_dims.iter() {
                                arg_grad = arg_grad.sum_keepdim(dim)?;
                            }

                            for _i in 0..left_dims {
                                arg_grad = arg_grad.squeeze(0)?
                            }
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.add(&arg_grad.broadcast_as(sum_grad.dims())?)?;
                        }
                        
                        _ => unimplemented!(),
                    }
                }
            }
        }

        Ok(grads)
    }

    pub fn sorted_nodes(&self) -> Vec<&Tensor<T>> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a, T: FloatDType>(
            node: &'a Tensor<T>,
            nodes: Vec<&'a Tensor<T>>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor<T>>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_leaf() {
                track_grad = true;
                nodes
            } else if node.dtype().is_int() {
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    | Op::Binary(lhs, rhs, _)
                    | Op::Matmul(lhs, rhs) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    
                    | Op::Unary(_node, UnaryOp::Ceil)
                    | Op::Unary(_node, UnaryOp::Floor)
                    | Op::Unary(_node, UnaryOp::Round)
                    | Op::Unary(_node, UnaryOp::Sign) => nodes,

                    | Op::BinaryScalar(node, _, _)
                    | Op::Broadcast(node)
                    | Op::Unary(node, _)
                    | Op::Reduce(node, _, _)
                    | Op::ReduceAll(node, _) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    fn broadcast_back(arg: &Tensor<T>, node: &Tensor<T>, reduced_dims: &[usize]) -> crate::Result<Tensor<T>> {
        if arg.rank() == node.rank() {
            node.broadcast_as(arg.shape())
        } else {
            node.reshape(reduced_dims)?.broadcast_as(arg.shape())
        }
    }    
}
