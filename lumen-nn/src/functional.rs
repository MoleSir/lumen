use lumen_core::{Dim, FloatDType, IntTensor, NumDType, Tensor, D};
use crate::{NnError, NnResult};

pub fn linear<T: NumDType>(input: &Tensor<T>, weight: &Tensor<T>, bias: Option<&Tensor<T>>) -> NnResult<Tensor<T>> {
    let x = match input.dims() {
        &[b1, b2, m, k] => {
            let (out_dim, in_dim) = weight.dims2()?;
            if input.is_contiguous() {
                let w = weight.transpose_last()?;
                input.reshape((b1 * b2 * m, k))?
                    .matmul(&w)?
                    .reshape((b1, b2, m, out_dim))?
            } else {
                let w = weight.broadcast_as((b1, b2, out_dim, in_dim))?;
                input.matmul(&w)?
            }
        }
        &[bsize, m, k] => {
            let (out_dim, in_dim) = weight.dims2()?;
            if input.is_contiguous() {
                let w = weight.transpose_last()?;
                input.reshape((bsize * m, k))?
                    .matmul(&w)?
                    .reshape((bsize, m, out_dim))?
            } else {
                let w = weight.broadcast_as((bsize, out_dim, in_dim))?;
                input.matmul(&w)?
            }
        }
        _ => {
            let w = weight.transpose_last()?;
            input.matmul(&w)?
        }
    };
    match bias {
        None => Ok(x),
        Some(bias) => {
            let out = x.broadcast_add(bias)?;
            Ok(out)
        }
    }
}

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use lumen_core::Tensor;
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]]).unwrap();
/// let a = lumen_nn::functional::softmax(&a, 1).unwrap();
/// ```
pub fn softmax<T: FloatDType, D: Dim>(xs: &Tensor<T>, dim: D) -> NnResult<Tensor<T>> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?; // (..., 1, ...)
    let diff = xs.broadcast_sub(&max)?;   // (..., D, ...)
    let num = diff.exp();                 // (..., D, ...)
    let den = num.sum_keepdim(dim)?;
    let out = num.broadcast_div(&den)?;
    Ok(out)
}

pub fn log_softmax<T: FloatDType, D: Dim>(xs: &Tensor<T>, dim: D) -> NnResult<Tensor<T>> {
    let dim = dim.to_index(xs.shape(), "log_softmax")?;
    let max = xs.max_keepdim(dim)?; // (..., 1, ...)
    let diff = xs.broadcast_sub(&max)?;   // (..., D, ...)
    //  log(sum(exp(x - max)))
    let log_sum_exp = diff.exp().sum_keepdim(dim)?.ln();
    // (x - max) - log_sum_exp
    let out = diff.broadcast_sub(&log_sum_exp)?;
    Ok(out)
}

pub fn silu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.silu())
}

pub fn swiglu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    let xs = xs.chunk(2, D::Minus1)?;
    Ok(xs[0].silu() * &xs[1])
}

pub fn sigmoid<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.sigmoid())
}

pub fn relu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.relu())
}

pub fn hard_sigmoid<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    let out = ((xs + T::from_f64(3.0)) / T::from_f64(6.0)).clamp(T::zero(), T::one())?;
    Ok(out)
}

pub fn leaky_relu<T: FloatDType>(xs: &Tensor<T>, negative_slope: T) -> NnResult<Tensor<T>> {
    Ok(xs.leaky_relu(negative_slope))
}

pub fn dropout<T: FloatDType>(xs: &Tensor<T>, drop_p: T) -> NnResult<Tensor<T>> {
    if drop_p < T::zero() || drop_p >= T::one() {
        Err(NnError::DropoutInvalid(drop_p.to_f64()))?;
    }

    let rand = Tensor::<T>::rand(T::zero(), T::one(), xs.shape())?;
    let scale = T::one() / (T::one() - drop_p);
    let mask = rand.ge(drop_p)?.to_dtype() * scale;
    Ok(xs * mask)
}

pub fn nll_loss<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
    let target = target.into();
    let gathered = input.gather(target, 1)?;
    let neg_loss = gathered.neg();
    let out = neg_loss.mean_all()?;
    Ok(out)
}

pub fn mse_loss<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> NnResult<Tensor<T>> {
    let out = input.sub(target)?.sqr().mean_all()?;
    Ok(out)
}

/// Computes the Cross Entropy Loss between input logits and target probabilities.
///
/// This function expects the `target` to be a probability distribution (e.g., one-hot vectors)
/// rather than class indices. It applies `log_softmax` on the input implicitly.
///
/// The result is averaged over the batch (reduction: mean).
///
/// ## Arguments
///
/// * `input` - The input tensor containing unnormalized scores (logits).
///   **Shape:** `(batch_size, num_classes)`
///
/// * `target` - The target tensor containing probabilities (0.0 to 1.0).
///   **Shape:** `(batch_size, num_classes)` (Must match `input` shape).
///
/// ## Returns
///
/// * A scalar Tensor representing the mean loss over the batch.
pub fn cross_entropy<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> NnResult<Tensor<T>> {
    let dim = 1; 
    let log_probs = log_softmax(input, dim)?;
    let weighted_log_probs = target.broadcast_mul(&log_probs)?;
    let sum_class = weighted_log_probs.sum_keepdim(dim)?;
    let loss = sum_class.neg();
    let out = loss.mean_all()?;
    Ok(out)
}

/// Computes the Cross Entropy Loss using integer class indices as targets.
///
/// This is the most common form of Cross Entropy used for classification.
/// It combines `log_softmax` and `nll_loss` (Negative Log Likelihood) for numerical stability.
///
/// The result is averaged over the batch (reduction: mean).
///
/// ## Arguments
///
/// * `input` - The input tensor containing unnormalized scores (logits).
///   **Shape:** `(batch_size, num_classes)`
///
/// * `target` - The target tensor containing class indices.
///   **Shape:** `(batch_size)` or `(batch_size, 1)`
///   Values must be integers in the range `[0, num_classes - 1]`.
///
/// ## Returns
///
/// * A scalar Tensor representing the mean loss over the batch.
pub fn cross_entropy_indices<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
    // Input: [samples, classes] (logits)
    // Target: [samples, 1] (probabilities, 0.0~1.0)
    let log_probs = log_softmax(input, 1)?;
    nll_loss(&log_probs, target)
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    #[test]
    fn test_nll_loss() {
        // Batch=2, Class=3
        // Sample 1: predict class 0 is high prob (-0.1)
        // Sample 2: predict class 2 is high prob (-0.2)
        let input = Tensor::new(&[
            [-0.1f32, -2.0, -3.0], 
            [-1.5, -0.5, -0.2]
        ]).unwrap();

        // Target: Sample 1 选 class 0, Sample 2 选 class 2
        let target = Tensor::new(&[0, 2]).unwrap().reshape((2, 1)).unwrap();

        let loss = crate::functional::nll_loss(&input, target).unwrap();

        // Calculation:
        // Sample 1 loss: -(-0.1) = 0.1
        // Sample 2 loss: -(-0.2) = 0.2
        // Mean: (0.1 + 0.2) / 2 = 0.15
        
        let val = loss.to_scalar().unwrap();
        assert!((val - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_soft_target() {
        // Input: Logits (unnormalized)
        let input = Tensor::new(&[
            [1.0f32, 2.0, 3.0], 
            [1.0, 2.0, 3.0]
        ]).unwrap();

        // Target: Probabilities (One-hot 模拟)
        // Sample 1: Class 2
        // Sample 2: Class 0
        let target = Tensor::new(&[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ]).unwrap();

        let loss = crate::functional::cross_entropy(&input, &target).unwrap();
        
        // Softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
        // LogSoftmax ≈ [-2.4076, -1.4076, -0.4076]
        
        // Sample 1 (Class 2): -1.0 * (-0.4076) = 0.4076
        // Sample 2 (Class 0): -1.0 * (-2.4076) = 2.4076
        // Mean = (0.4076 + 2.4076) / 2 = 1.4076
        
        let val = loss.to_scalar().unwrap();
        println!("{}", val);
        assert!((val - 1.4076).abs() < 1e-3);
    }
}