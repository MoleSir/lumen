use crate::rng;
use crate::error::TensorError;
use crate::{Tensor, Range, Shape};
use crate::backward::{Backward, BinaryTensorBackward, MultiTensorBackward, UnaryTensorBackward};
use crate::utils;
use anyhow::Result;

/// Add `lhs`` and `rhs`.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap();
/// let y = op::add(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build([1. + 3., 2. + 1., 3. + 5., 4. + 0.], [2, 2]).unwrap()));
/// 
/// let a = Tensor::build([1., 2.], [2]).unwrap();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap();
/// let y = op::add(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build([1. + 3., 2. + 1., 1. + 5., 2. + 0.], [2, 2]).unwrap()));
/// ```
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.dim_size() == rhs.dim_size() {
        // `lhs` and `rhs` have the same dim size, so they must has same shape to add
        lhs.check_same_shape(rhs)?;

        let result = lhs.iter().zip(rhs.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let shape = lhs.shape().clone();
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        Ok(Tensor::new(result, shape, Backward::Binary(BinaryTensorBackward::Add(lhs.clone(), rhs.clone())), requires_grad))
    } else {
        // Try broadcast, the short dim tensor should has the same shape in low dim's of long tensor
        let (long, short) = if lhs.dim_size() > rhs.dim_size() { (lhs, rhs) } else { (rhs, lhs) };

        // Compare low dim
        if !short.shape().iter().rev().zip(long.shape().iter().rev())
            .all(|(ss, ls)| ss == ls) 
        {
            return Err(TensorError::DifferentShape)?;
        }

        // Create a new zero tensor
        let shape: Shape = long.shape().clone().into();
        let data = vec![0.; shape.element_size()];
        let operate = Backward::Binary(BinaryTensorBackward::BroadcastAdd(lhs.clone(), rhs.clone()));
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let dst = Tensor::new(data, shape, operate, requires_grad);

        // Add all slice of long tensor to short
        let high_dim = long.dim_size() - short.dim_size();
        for ranges in utils::generate_coordinates(&long.shape()[..high_dim]) {
            let mut ranges: Vec<_> = ranges.into_iter().map(|r| rng!(r)).collect();
            for _ in 0..short.dim_size() {
                ranges.push(rng!(:));
            }

            let dst_view = dst.slice(&ranges).unwrap();
            let long_view = long.slice(&ranges).unwrap();
            utils::add_(&dst_view, &long_view, short);
        } 
        
        Ok(dst) 
    }
}

/// Sub `lhs`` and `rhs`.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap();
/// let y = op::sub(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build([1. - 3., 2. - 1., 3. - 5., 4. - 0.], [2, 2]).unwrap()));
/// ```
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.check_same_shape(rhs)?;

    let result = lhs.iter().zip(rhs.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();

    let shape = lhs.shape().clone();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    Ok(Tensor::new(result, shape, Backward::Binary(BinaryTensorBackward::Sub(lhs.clone(), rhs.clone())), requires_grad))
}

/// Mul `lhs`` and `rhs`.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let b = Tensor::build([3., 1., 5., 0.], [2, 2]).unwrap();
/// let y = op::mul(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build([1. * 3., 2. * 1., 3. * 5., 4. * - 0.], [2, 2]).unwrap()));
/// ```
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.check_same_shape(rhs)?;

    let result = lhs.iter().zip(rhs.iter())
        .map(|(a, b)| a * b)
        .collect::<Vec<_>>();

    let shape = lhs.shape().clone();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    Ok(Tensor::new(result, shape, Backward::Binary(BinaryTensorBackward::Mul(lhs.clone(), rhs.clone())), requires_grad))
}

/// Div `lhs`` and `rhs`.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let b = Tensor::build([3., 1., 5., 7.], [2, 2]).unwrap();
/// let y = op::div(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build([1. / 3., 2. / 1., 3. / 5., 4. / 7.], [2, 2]).unwrap()));
/// ```
pub fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    lhs.check_same_shape(rhs)?;

    let result = lhs.iter().zip(rhs.iter())
        .map(|(a, b)| a / b)
        .collect::<Vec<_>>();

    let shape = lhs.shape().clone();
    let requires_grad = lhs.requires_grad() || rhs.requires_grad();

    Ok(Tensor::new(result, shape, Backward::Binary(BinaryTensorBackward::Div(lhs.clone(), rhs.clone())), requires_grad))
}

/// Neg all element
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::ones([2, 3]);
/// let y = op::neg(&a);
/// assert!(y.allclose(&Tensor::build([-1., -1., -1., -1., -1., -1.], [2, 3]).unwrap()));
/// ```
pub fn neg(tensor: &Tensor) -> Tensor {
    mul(&tensor.fill_like(-1.), tensor).unwrap()
}

/// Matrix dot operate
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let a = Tensor::build(vec![1., 2., 3., 2., 4., 7.], [2, 3]).unwrap();
/// let b = Tensor::build(vec![3., 4., 5., 6., 0., 1.], [3, 2]).unwrap();
/// let y = op::matmul(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build(vec![13., 19., 26., 39.], [2, 2]).unwrap()));
/// 
/// let a = Tensor::build(vec![1., 2., 3., 2., 4., 7., 1., 2., 3., 2., 4., 7.], [2, 2, 3]).unwrap();
/// let b = Tensor::build(vec![3., 4., 5., 6., 0., 1., 3., 4., 5., 6., 0., 1.], [2, 3, 2]).unwrap();
/// let y = op::matmul(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build(vec![13., 19., 26., 39., 13., 19., 26., 39.], [2, 2, 2]).unwrap()));
/// 
/// let a = Tensor::build(vec![1., 2., 3., 2., 4., 7., 1., 2., 3., 2., 4., 7.], [1, 2, 2, 3]).unwrap();
/// let b = Tensor::build(vec![3., 4., 5., 6., 0., 1., 3., 4., 5., 6., 0., 1.], [1, 2, 3, 2]).unwrap();
/// let y = op::matmul(&a, &b).unwrap();
/// assert!(y.allclose(&Tensor::build(vec![13., 19., 26., 39., 13., 19., 26., 39.], [1, 2, 2, 2]).unwrap()));
/// ```
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.dim_size() != rhs.dim_size() {
        return Err(TensorError::DimensionsUnmatch)?;
    }
    
    // self and rhs has same dim size
    match lhs.dim_size() {
        0 | 1 => Err(TensorError::MatMulShapeError)?,
        2 => {
            let (r1, c1) = (lhs.shape()[0], lhs.shape()[1]);
            let (r2, c2) = (rhs.shape()[0], rhs.shape()[1]);
            
            if c1 != r2 {
                return Err(TensorError::MatMulShapeError)?;
            }

            let operate = Backward::Binary(BinaryTensorBackward::MatMul(lhs.clone(), rhs.clone()));
            let dst = Tensor::new(vec![0.; r1 * c2], [r1, c2], operate, lhs.requires_grad() || rhs.requires_grad());

            utils::matmul_(&dst, lhs, rhs);

            Ok(dst)
        }
        dim_size => {
            let shape1 = lhs.shape().iter().cloned().collect::<Vec<_>>();
            let shape2 = rhs.shape().iter().cloned().collect::<Vec<_>>();

            // shape equal except last two dim
            let high_dim_size = dim_size - 2;
            assert!(high_dim_size != 0);

            if shape1[..high_dim_size] != shape2[..high_dim_size] {
                return Err(TensorError::MatMulShapeError)?;
            }

            let (r1, c1) = (shape1[high_dim_size], shape1[high_dim_size + 1]);
            let (r2, c2) = (shape2[high_dim_size], shape2[high_dim_size + 1]);

            if c1 != r2 {
                return Err(TensorError::MatMulShapeError)?;
            }

            let mut dst_shape = shape1[..high_dim_size].to_vec();
            dst_shape.push(r1);
            dst_shape.push(c2);
            let shape: Shape = dst_shape.into();
            let data = vec![0.; shape.element_size()];
            let operate = Backward::Binary(BinaryTensorBackward::MatMul(lhs.clone(), rhs.clone()));
            let requires_grad= lhs.requires_grad() || rhs.requires_grad();
            let dst =  Tensor::new(data, shape, operate, requires_grad);
            
            for ranges in utils::generate_coordinates(&shape1[..high_dim_size]).into_iter()
                .map(|idx| {
                    let mut idx: Vec<_> = idx.into_iter().map(|i| Range::index(i)).collect();
                    idx.push(rng!(:));
                    idx.push(rng!(:));
                    idx
                })
            {
                let view_dst = dst.slice(&ranges).unwrap();
                let view_lhs = lhs.slice(&ranges).unwrap();
                let view_rhs = rhs.slice(&ranges).unwrap();
                utils::matmul_(&view_dst, &view_lhs, &view_rhs);
            }

            Ok(dst)
        }
    }
}

/// Returns a new tensor with the exponential of the elements of the input tensor input.
/// 
/// y{i} = exp^{x{i}}
/// 
/// # Example 
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::exp(&x);
/// assert!(y.allclose(&Tensor::build(vec![2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236], [2, 2]).unwrap()));
/// ```
pub fn exp(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|v| v.exp())
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Exp(tensor.clone())), tensor.requires_grad())
}

/// Takes the power of each element in input with exponent and returns a tensor with the result.
/// 
/// y{i} = x{i}^{exponent}
/// 
/// # Example 
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::pow(&x, 1.5);
/// assert!(y.allclose(&Tensor::build(vec![1.0, 2.8284271247461903, 5.196152422706632, 8.0], [2, 2]).unwrap()));
/// ```
pub fn pow(tensor: &Tensor, exponent: f64) -> Tensor {
    let result = tensor.iter()
        .map(|v| v.powf(exponent))
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Pow(tensor.clone(), exponent)), tensor.requires_grad())
}

/// Returns a new tensor with the hyperbolic tangent of the elements of input
/// 
/// y{i} = tanh(x{i})
/// 
/// # Example 
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::tanh(&x);
/// assert!(y.allclose(&Tensor::build(vec![0.7615941559557649, 0.9640275800758169, 0.9950547536867305, 0.999329299739067], [2, 2]).unwrap()));
/// ```
pub fn tanh(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|v| v.tanh())
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Tanh(tensor.clone())), tensor.requires_grad())
}

/// Returns a new tensor with the sigmoidof of the input tensor input.
/// 
/// y{i} = sigmoid(x{i}) = 1 / (1 + exp^{-x{i}})
/// 
/// # Example 
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::sigmoid(&x);
/// assert!(y.allclose(&Tensor::build(vec![0.7310585786300049, 0.8807970779778823, 0.9525741268224334, 0.9820137900379085], [2, 2]).unwrap()));
/// ```
pub fn sigmoid(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|x| 1. / (1. + (-x).exp()))
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Sigmoid(tensor.clone())), tensor.requires_grad())
}

/// Return a new tensor with the abs of the input tensor
///
/// y_{i} = \text{abs}(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::build(vec![1., -2., -3., 4.], [2, 2]).unwrap();
/// let y = op::abs(&x);
/// assert!(y.allclose(&Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap()));
/// ```
pub fn abs(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|x| x.abs())
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Abs(tensor.clone())), tensor.requires_grad())
}

/// Returns a new tensor with the sqrt of the input tensor input.
/// 
/// y_{i} = sqrt(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::sqrt(&x);
/// assert!(y.allclose(&Tensor::build(vec![1f64.sqrt(), 2f64.sqrt(), 3f64.sqrt(), 4f64.sqrt()], [2, 2]).unwrap()));
/// ```
pub fn sqrt(tensor: &Tensor) -> Tensor {
    pow(tensor, 0.5)
}

/// Returns a new tensor with the log of the input tensor input.
///
/// y_{i} = log_{base}(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::log(&x, 2.);
/// assert!(y.allclose(&Tensor::build(vec![1f64.log(2.), 2f64.log(2.), 3f64.log(2.), 4f64.log(2.)], [2, 2]).unwrap()));
/// ```
pub fn log(tensor: &Tensor, base: f64) -> Tensor {
    let result = tensor.iter()
        .map(|x| x.log(base))
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Log(tensor.clone(), base)), tensor.requires_grad())
}

/// Returns a new tensor with the ln of the input tensor input.
///
/// y_{i} = ln(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([[1, 2], [3, 4]]);
/// let y = op::ln(&x);
/// assert!(y.allclose(&Tensor::build(vec![1f64.ln(), 2f64.ln(), 3f64.ln(), 4f64.ln()], [2, 2]).unwrap()));
/// ```
pub fn ln(tensor: &Tensor) -> Tensor {
    log(tensor, 1f64.exp())
}

/// Returns a new tensor with the ln of the input tensor input.
///
/// y_{i} = ln(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::build(vec![-1., 2., 3., -4.], [2, 2]).unwrap();
/// let y = op::relu(&x);
/// assert!(y.allclose(&Tensor::build(vec![0., 2., 3., 0.], [2, 2]).unwrap()));
/// ```
pub fn relu(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|x| x.max(0.))
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::ReLU(tensor.clone())), tensor.requires_grad())
}

/// Returns a new tensor with the sin of the input tensor input.
///
/// y_{i} = sin(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([1, 2, 3, 4, 5]);
/// let y = op::sin(&x);
/// assert!(y.allclose(&Tensor::from([1f64.sin(), 2f64.sin(), 3f64.sin(), 4f64.sin(), 5f64.sin()])));
/// ```
pub fn sin(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|x| x.sin())
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Sin(tensor.clone())), tensor.requires_grad())
}

/// Returns a new tensor with the cos of the input tensor input.
///
/// y_{i} = cos(x_{i})
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let x = Tensor::from([1, 2, 3, 4, 5]);
/// let y = op::cos(&x);
/// assert!(y.allclose(&Tensor::from([1f64.cos(), 2f64.cos(), 3f64.cos(), 4f64.cos(), 5f64.cos()])));
/// ```
pub fn cos(tensor: &Tensor) -> Tensor {
    let result = tensor.iter()
        .map(|x| x.cos())
        .collect::<Vec<_>>();

    let shape = tensor.shape().clone();
    Tensor::new(result, shape, Backward::Unary(UnaryTensorBackward::Cos(tensor.clone())), tensor.requires_grad())
}

/// Concatenates a sequence of tensors along a new dimension.
/// 
/// All tensors need to be of the same size.
/// 
/// # Example
/// 
/// ```rust
/// use lumen_core::*;
/// let t = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let y = op::stack(&[t.clone(), t.clone()]).unwrap();
/// assert!(y.allclose(&Tensor::build([1., 2., 3., 4., 1., 2., 3., 4.], [2, 2, 2]).unwrap()));
/// 
/// let t1 = Tensor::build([1., 2., 3., 4.], [2, 2]).unwrap();
/// let t2 = Tensor::build([5., 2., 1., 2.], [2, 2]).unwrap();
/// let y = op::stack(&[t1.clone(), t2.clone(), t1.clone()]).unwrap();
/// assert!(y.allclose(&Tensor::build([1., 2., 3., 4., 5., 2., 1., 2., 1., 2., 3., 4.], [3, 2, 2]).unwrap()));
/// ```
pub fn stack(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.len() == 0 {
        return Err(TensorError::DifferentShape)?;
    }
    // Check size
    let shape = tensors[0].shape();
    if !tensors.iter().skip(1)
        .all(|t| *t.shape() == *shape)
    {
        return Err(TensorError::DifferentShape)?;
    } else {
        let dst_shape = [tensors.len()].iter()
            .chain(shape.iter()).cloned().collect::<Vec<_>>();
        
        let mut pre_tensors = Vec::new();
        let dst = Tensor::zeros(dst_shape);
        for (i, src_tensor) in tensors.iter().enumerate() {
            let ranges: Vec<_> = [rng!(i)].into_iter()
                .chain(vec![rng!(:); src_tensor.dim_size()].into_iter()).collect();
            
            let dst_view = dst.slice(&ranges).unwrap();
            copy(&dst_view, src_tensor).unwrap();

            if src_tensor.requires_grad() {
                pre_tensors.push(src_tensor.clone());
            }
        }

        let dst = if !pre_tensors.is_empty() {
            dst.with_operate(Backward::Multi(MultiTensorBackward::Stack(pre_tensors)));
            dst.require_grad()
        } else {
            dst
        };

        Ok(dst)
    }
}

/// Copy `src` to `dst`
/// 
/// Two tensors need to be of the same size.
/// 
pub fn copy(dst: &Tensor, src: &Tensor) -> Result<()> {
    if *dst.shape() != *src.shape() {
        Err(TensorError::DifferentShape)?
    } else {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d = s;
        } 
        Ok(())
    }
}

