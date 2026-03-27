use std::collections::HashMap;
use crate::{FloatDType, Tensor, TensorId};

use super::{BinaryOp, GradStore, Op, ReduceOp, UnaryOp};

impl<T: FloatDType> Tensor<T> {

    pub fn backward(&self) -> crate::Result<GradStore<T>> {
        let _guard = crate::NoGradGuard::new();

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
                            lhs_sum_grad.impl_add_(&grad)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Sub) => {
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&grad)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_sub_(&grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Mul) => {
                            let lhs_grad = grad.mul(rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?;
                            
                            let rhs_grad = grad.mul(lhs)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&rhs_grad)?;
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Div) => {
                            let lhs_grad = grad.div(rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?; 
                            
                            let rhs_grad = grad.mul(lhs)?.div(&rhs.sqr()?)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_sub_(&rhs_grad)?; 
                        }
                        Op::Binary(lhs, rhs, BinaryOp::Minimum)
                        | Op::Binary(lhs, rhs, BinaryOp::Maximum) => {
                            let mask_lhs = (*node).eq(lhs)?.cast()?;
                            let mask_rhs = (*node).eq(rhs)?.cast()?;
    
                            // If both masks are 1 one the same point, we want to scale the
                            // gradient by 0.5 rather than 1.
                            let lhs_grad = mask_lhs.mul(&grad)?.div(&(&mask_rhs + T::one()))?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?;
    
                            let rhs_grad = mask_rhs.mul(&grad)?.div(&(&mask_lhs + T::one()))?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&rhs_grad)?;
                        }
                        
                        //=========================================================================================//
                        //           BinaryScalarRhs
                        //=========================================================================================//
                        Op::BinaryScalarRhs(lhs, _, BinaryOp::Add) => {
                            // y = x + c => dy/dx = 1
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&grad)?;
                        }
                        Op::BinaryScalarRhs(lhs, _, BinaryOp::Sub) => {
                            // y = x - c => dy/dx = 1
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&grad)?;
                        }
                        Op::BinaryScalarRhs(lhs, rhs, BinaryOp::Mul) => {
                            // y = x * c => dy/dx = c
                            let lhs_grad = grad.mul_scalar(*rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?;
                        }
                        Op::BinaryScalarRhs(lhs, rhs, BinaryOp::Div) => {
                            // y = x / c => dy/dx = 1/c
                            let lhs_grad = grad.div_scalar(*rhs)?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?;
                        }
                        Op::BinaryScalarRhs(lhs, rhs, BinaryOp::Maximum) |
                        Op::BinaryScalarRhs(lhs, rhs, BinaryOp::Minimum) => {
                            let mask_lhs = (*node).eq(lhs)?.cast()?;                            
                            let mask_rhs = (*node).eq(*rhs)?.cast()?;
                            let lhs_grad = mask_lhs.mul(&grad)?.div(&(&mask_rhs + T::one()))?;
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.impl_add_(&lhs_grad)?;
                        }

                        //=========================================================================================//
                        //           BinaryScalarLhs
                        //=========================================================================================//
                        Op::BinaryScalarLhs(_, rhs, BinaryOp::Add) => {
                            // y = c + x => dy/dx = 1
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&grad)?;
                        }
                        Op::BinaryScalarLhs(_, rhs, BinaryOp::Sub) => {
                            // y = c - x => dy/dx = -1
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_sub_(&grad)?; 
                        }
                        Op::BinaryScalarLhs(lhs, rhs, BinaryOp::Mul) => {
                            // y = c * x => dy/dx = c
                            let rhs_grad = grad.mul_scalar(*lhs)?;
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&rhs_grad)?;
                        }
                        Op::BinaryScalarLhs(lhs, rhs, BinaryOp::Div) => {
                            // y = c / x = c * x^(-1)
                            // dy/dx = -c * x^(-2) = -c / (x^2)
                            // grad_input = grad * (-c / x^2)
                            let numerator = grad.mul_scalar(-*lhs)?;                            
                            let denominator = rhs.mul(rhs)?;                          
                            let rhs_grad = numerator.div(&denominator)?;
                            
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&rhs_grad)?;
                        }
                        Op::BinaryScalarLhs(lhs, rhs, BinaryOp::Maximum) |
                        Op::BinaryScalarLhs(lhs, rhs, BinaryOp::Minimum) => {
                            let mask_lhs = (*node).eq(*lhs)?.cast()?;
                            let mask_rhs = (*node).eq(rhs)?.cast()?;
                            let rhs_grad = mask_rhs.mul(&grad)?.div(&(&mask_lhs + T::one()))?;                            
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.impl_add_(&rhs_grad)?;
                        }

                        //=========================================================================================//
                        //           Unary
                        //=========================================================================================//
                        Op::Unary(_, UnaryOp::Ceil) => Err(crate::Error::BackwardNotSupported("ceil"))?,
                        Op::Unary(_, UnaryOp::Floor) => Err(crate::Error::BackwardNotSupported("floor"))?,
                        Op::Unary(_, UnaryOp::Round) => Err(crate::Error::BackwardNotSupported("round"))?,                        
                        Op::Unary(_, UnaryOp::Sign) => Err(crate::Error::BackwardNotSupported("sign"))?,
                        Op::Unary(arg, UnaryOp::Exp) => {
                            let arg_grad = grads.or_insert(arg)?;
                            arg_grad.impl_add_mul_(&grad, node)?;
                        }
                        Op::Unary(arg, UnaryOp::Ln) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_div_(&grad, arg)?;
                        }
                        Op::Unary(arg, UnaryOp::Sin) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, T::add, |a, b| a * b.cos(), "fused in sin backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Cos) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, T::sub, |a, b| a * b.sin(), "fused in Cos backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Tanh) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &node, T::sub, |a, b| a * b.sqr() - T::ONE, "fused in Tanh backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Sqr) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &arg, &grad, 
                                |a, b| a + T::two() * b, T::mul, "fused in Sqr backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Sqrt) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &node, 
                                |a, b| a + T::half() * b, T::div, "fused in Sqrt backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Abs) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, |a, b| a * ( if b >= T::ZERO {T::ONE} else {T::ONE.neg()} ) , "fused in Abs backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Neg) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_sub_(&grad)?;
                        }
                        Op::Unary(arg, UnaryOp::Recip) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::sub, |a, b| a / b.sqr(), "fused in Recip backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Gelu) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let c1 = T::from_f64(0.0356774);
                            let c2 = T::from_f64(0.797885);
                            let c3 = T::from_f64(0.0535161);
                            let c4 = T::from_f64(0.398942);
                        
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                move |g, x| {
                                    let x3 = x * x.sqr();
                                    let inner = x3 * c1 + x * c2;
                                    let tanh_val = inner.tanh();
                                    let dt = x3 * c3 + x * c4;
                                    let gelu_grad = tanh_val * T::half() + dt * (T::one() - tanh_val.sqr()) + T::half();
                                    g * gelu_grad
                                }, 
                                "fused in Gelu backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Erf) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let scale = T::two() / T::pi().sqrt();
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                move |g, x| g * x.sqr().neg().exp() * scale, 
                                "fused in Erf backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::GeluErf) => {
                            let sum_grad = grads.or_insert(arg)?;
                            let c1 = T::from_f64(0.398942);
                            let sqrt_two = T::two().sqrt();
                        
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                move |g, x| {
                                    let neg_half_square = x.sqr().neg() / T::two();
                                    let scaled_exp_arg = c1 * neg_half_square.exp() * x;
                                    let arg_scaled_sqrt = x / sqrt_two;
                                    let erf_scaled_sqrt = arg_scaled_sqrt.erf() / T::two();
                                    let gelu_erf_grad = scaled_exp_arg + erf_scaled_sqrt + T::half();
                                    g * gelu_erf_grad
                                }, 
                                "fused in GeluErf backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Relu) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                |g, x| if x >= T::zero() { g } else { T::zero() }, 
                                "fused in Relu backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Silu) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                |g, x| {
                                    let sigmoid = (x.neg().exp() + T::one()).recip();
                                    let silu_val = x * sigmoid;
                                    let silu_grad = sigmoid * (T::one() - silu_val) + silu_val;
                                    g * silu_grad
                                }, 
                                "fused in Silu backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Sigmoid) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &node, 
                                T::add, 
                                |g, y| g * y * (T::one() - y), 
                                "fused in Sigmoid backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::LeakyRelu(negative_slope)) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                move |g, x| if x >= T::zero() { g } else { g * *negative_slope }, 
                                "fused in LeakyRelu backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Pow(e)) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_2arg_inplace_op(
                                &grad, &arg, 
                                T::add, 
                                move |g, x| g * x.powf(*e - T::one()) * *e, 
                                "fused in Pow backward"
                            )?;
                        }
                        Op::Unary(arg, UnaryOp::Affine { mul, add: _ }) => {
                            // y = x * mul + add
                            // dy = mul
                            let arg_grad = grad.mul_scalar(*mul)?;
                            let lhs_sum_grad = grads.or_insert(arg)?;
                            lhs_sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Matmul
                        //=========================================================================================//
                        Op::Matmul(lhs, rhs) => {    
                            let lhs_sum_grad = grads.or_insert(lhs)?;
                            lhs_sum_grad.add_matmul_(&grad, &rhs.transpose_last()?)?;
    
                            let rhs_sum_grad = grads.or_insert(rhs)?;
                            rhs_sum_grad.add_matmul_(&lhs.transpose_last()?, &grad)?;
                        }

                        //=========================================================================================//
                        //           Reduce
                        //=========================================================================================//
                        Op::Reduce(arg, ReduceOp::Sum, reduced_dims) => {
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&grad)?;
                        }
                        Op::Reduce(arg, ReduceOp::Max, reduced_dims) => {
                            let node = Self::broadcast_back(arg, node, reduced_dims)?;
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let grad = node.eq(arg)?.cast()?.mul(&grad)?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&grad.broadcast_as(sum_grad.dims())?)?;
                        }
                        Op::Reduce(arg, ReduceOp::Min, reduced_dims) => {
                            let node = Self::broadcast_back(arg, node, reduced_dims)?;
                            let grad = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let grad = node.eq(arg)?.cast()?.mul(&grad)?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&grad.broadcast_as(sum_grad.dims())?)?;
                        }
                        Op::Reduce(arg, ReduceOp::Mean, reduced_dims) => {
                            let grad_output = Self::broadcast_back(arg, &grad, reduced_dims)?;
                            let n = arg.element_count() / node.element_count();
                            let scale = T::from_usize(n);
                            
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.fused_1arg_inplace_op(
                                &grad_output, T::add, |v| v / scale , "fused in Mean backward" 
                            )?;
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
                            sum_grad.impl_add_(&arg_grad.broadcast_as(sum_grad.dims())?)?;
                        }

                        //=========================================================================================//
                        //           Narrow
                        //=========================================================================================//
                        &Op::Narrow(ref arg, dim, start_idx, len) => {
                            let arg_dims = arg.dims();
                            let left_pad = if start_idx == 0 {
                                None
                            } else {
                                let mut dims = arg_dims.to_vec();
                                dims[dim] = start_idx;
                                Some(Tensor::zeros(dims)?)
                            };
                            let right_pad = arg_dims[dim] - start_idx - len;
                            let right_pad = if right_pad == 0 {
                                None
                            } else {
                                let mut dims = arg_dims.to_vec();
                                dims[dim] = right_pad;
                                Some(Tensor::zeros(dims)?)
                            };
                            let arg_grad = match (left_pad, right_pad) {
                                (None, None) => grad,
                                (Some(l), None) => Tensor::cat(&[&l, &grad], dim)?,
                                (None, Some(r)) => Tensor::cat(&[&grad, &r], dim)?,
                                (Some(l), Some(r)) => Tensor::cat(&[&l, &grad, &r], dim)?,
                            };
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Slice
                        //=========================================================================================//
                        &Op::Slice(ref arg, dim, start, _end, step) => {
                            let arg_dims = arg.dims();
                            
                            let body_grad = if step == 1 {
                                grad
                            } else {
                                let grad_len = grad.dims()[dim];
                                let span_len = if grad_len > 0 { (grad_len - 1) * step + 1 } else { 0 };
                                
                                let mut unsqueezed_shape = grad.dims().to_vec();
                                unsqueezed_shape.insert(dim + 1, 1);
                                let grad_unsqueezed = grad.reshape(&unsqueezed_shape)?;
                                
                                let mut zeros_shape = unsqueezed_shape.clone();
                                zeros_shape[dim + 1] = step - 1;
                                let zeros_gap = Tensor::zeros(zeros_shape)?;
                                
                                let dilated = Tensor::cat(&[&grad_unsqueezed, &zeros_gap], dim + 1)?;
                                
                                let mut flattened_shape = grad.dims().to_vec();
                                flattened_shape[dim] = grad_len * step;
                                let flattened = dilated.reshape(flattened_shape)?;
                                
                                flattened.narrow(dim, 0, span_len)?
                            };
                        
                            let body_len = body_grad.dims()[dim];
                            
                            let left_pad = if start == 0 {
                                None
                            } else {
                                let mut dims = arg_dims.to_vec();
                                dims[dim] = start;
                                Some(Tensor::zeros(dims)?)
                            };
                        
                            let right_pad_len = arg_dims[dim] - start - body_len;
                            let right_pad = if right_pad_len == 0 {
                                None
                            } else {
                                let mut dims = arg_dims.to_vec();
                                dims[dim] = right_pad_len;
                                Some(Tensor::zeros(dims)?)
                            };
                        
                            let arg_grad = match (left_pad, right_pad) {
                                (None, None) => body_grad,
                                (Some(l), None) => Tensor::cat(&[&l, &body_grad], dim)?,
                                (None, Some(r)) => Tensor::cat(&[&body_grad, &r], dim)?,
                                (Some(l), Some(r)) => Tensor::cat(&[&l, &body_grad, &r], dim)?,
                            };
                        
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Reshape
                        //=========================================================================================//
                        Op::Reshape(arg) => {
                            let arg_grad = grad.reshape(arg.dims())?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Transpose
                        //=========================================================================================//
                        Op::Transpose(arg, dim1, dim2) => {
                            let arg_grad = grad.transpose(*dim1, *dim2)?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Permute
                        //=========================================================================================//
                        Op::Permute(arg, dims) => {
                            let mut inv_dims = vec![0; dims.len()];
                            for (i, &dim_idx) in dims.iter().enumerate() {
                                inv_dims[dim_idx] = i
                            }
                            let arg_grad = grad.permute(inv_dims)?;
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&arg_grad)?;
                        }

                        //=========================================================================================//
                        //           Cat
                        //=========================================================================================//
                        Op::Cat(args, dim) => {
                            let mut start_idx = 0;
                            for arg in args {
                                let len = arg.dims()[*dim];
                                let arg_grad = grad.narrow(*dim, start_idx, len)?;
                                let sum_grad = grads.or_insert(arg)?;
                                sum_grad.impl_add_(&arg_grad)?;
                                start_idx += len;
                            }
                        }

                        //=========================================================================================//
                        //           Copy
                        //=========================================================================================//
                        Op::Copy(arg) => {
                            let sum_grad = grads.or_insert(arg)?;
                            sum_grad.impl_add_(&grad)?;
                        }

                        //=========================================================================================//
                        //           IfElse
                        //=========================================================================================//
                        Op::IfElse(mask, tv, fv) => {
                            if let Some(tv) = tv {
                                let masked_grad = mask.if_else(&grad, T::zero())?;
                                let sum_grad = grads.or_insert(tv)?;
                                sum_grad.impl_add_(&masked_grad)?;
                            }

                            if let Some(fv) = fv {
                                let masked_grad = mask.if_else(T::zero(), &grad)?;
                                let sum_grad = grads.or_insert(fv)?;
                                sum_grad.impl_add_(&masked_grad)?;
                            }
                        }

                        //=========================================================================================//
                        //           IndexSelect
                        //=========================================================================================//
                        Op::IndexSelect(arg, indexes, dim) => {
                            let sum_grad = grads.or_insert(arg)?;
                            *sum_grad = sum_grad.index_add(indexes.clone(), &grad, *dim)?;
                        }

                        //=========================================================================================//
                        //           IndexAdd
                        //=========================================================================================//
                        Op::IndexAdd(init, indexes, src, dim) => {
                            let init_sum_grad = grads.or_insert(init)?;
                            *init_sum_grad = init_sum_grad.add(&grad)?;
    
                            let src_grad = grad.index_select(indexes.clone(), *dim)?;
                            let src_sum_grad = grads.or_insert(src)?;
                            *src_sum_grad = src_sum_grad.add(&src_grad)?;
                        }

                        //=========================================================================================//
                        //           IndexAdd
                        //=========================================================================================//
                        #[allow(unused)]
                        Op::ScatterAdd(init, indexes, src, dim) => {
                            unimplemented!()
                        }

                        //=========================================================================================//
                        //           Gather
                        //=========================================================================================//
                        Op::Gather(arg, indexes, dim) => {
                            let arg_grad = grads.or_insert(arg)?;
                            *arg_grad = arg_grad.scatter_add(indexes.clone(), &grad, *dim)?;
                        }
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
                    | Op::Matmul(lhs, rhs) 
                    | Op::IfElse(_, Some(lhs), Some(rhs))
                    | Op::IndexAdd(lhs, _, rhs, _)
                    | Op::ScatterAdd(lhs, _, rhs, _)
                    => {
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

                    | Op::IfElse(_, None, None) => nodes,

                    | Op::BinaryScalarLhs(_, node, _)
                    | Op::BinaryScalarRhs(node, _, _)
                    | Op::Broadcast(node)
                    | Op::Unary(node, _)
                    | Op::Reduce(node, _, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Slice(node, _, _, _, _)
                    | Op::Reshape(node)
                    | Op::Transpose(node, _, _)
                    | Op::Permute(node, _)
                    | Op::Copy(node) 
                    | Op::Gather(node, _, _)
                    | Op::IndexSelect(node, _, _)
                    | Op::IfElse(_, Some(node), None)
                    | Op::IfElse(_, None, Some(node)) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }

                    | Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
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
