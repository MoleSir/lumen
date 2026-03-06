use lumen_core::{Dim, FloatDType, IntTensor, NumDType, Tensor, WithDType};
use thiserrorctx::Context;
use crate::{NnError, NnResult};

// ============================================================================= //
//                         Common 
// ============================================================================= //

/// Applies a linear transformation to the incoming data: $y = xA^T + b$.
///
/// This function handles broadcasting for inputs with more than 2 dimensions,
/// commonly used in Transformer blocks (e.g., `[batch, heads, seq, dim]`).
///
/// ## Arguments
///
/// * `input` - Input tensor of shape `(..., in_features)`.
/// * `weight` - Weights of shape `(out_features, in_features)`.
/// * `bias` - Optional bias of shape `(out_features)`.
///
/// ## Returns
///
/// * A Tensor with shape `(..., out_features)`.
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

/// Applies the Softmax function to an n-dimensional input Tensor.
///
/// Rescales elements so that they lie in the range [0, 1] and sum to 1.
///
/// ## Arguments
///
/// * `xs` - Input tensor.
/// * `dim` - A dimension along which Softmax will be computed.
///
/// ## Returns
///
/// * A Tensor with the same dimension and shape as the input with values in the range [0, 1].
///
/// ## Example
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
    let num = diff.exp()?;  // (..., D, ...)
    let den = num.sum_keepdim(dim)?;
    let out = num.broadcast_div(&den)?;
    Ok(out)
}

/// Applies the LogSoftmax function to an n-dimensional input Tensor.
///
/// Mathematically equivalent to `log(softmax(x))`, but implemented separately for numerical stability.
///
/// ## Arguments
///
/// * `xs` - Input tensor.
/// * `dim` - A dimension along which LogSoftmax will be computed.
///
/// ## Returns
///
/// * A Tensor of the same shape as `xs` containing the log probabilities.
pub fn log_softmax<T: FloatDType, D: Dim>(xs: &Tensor<T>, dim: D) -> NnResult<Tensor<T>> {
    let dim = dim.to_index(xs.shape(), "log_softmax")?;
    let max = xs.max_keepdim(dim)?; // (..., 1, ...)
    let diff = xs.broadcast_sub(&max)?;   // (..., D, ...)
    //  log(sum(exp(x - max)))
    let log_sum_exp = diff.exp()?.sum_keepdim(dim)?.ln()?;
    // (x - max) - log_sum_exp
    let out = diff.broadcast_sub(&log_sum_exp)?;
    Ok(out)
}

/// Randomly zeroes some of the elements of the input tensor with probability `drop_p`.
///
/// The output is scaled by a factor of $\frac{1}{1 - p}$ during training (Inverted Dropout).
///
/// ## Arguments
///
/// * `xs` - Input tensor.
/// * `drop_p` - The probability of an element to be zeroed (must be between 0.0 and 1.0).
///
/// ## Returns
///
/// * A Tensor of the same shape as `xs`.
pub fn dropout<T: FloatDType>(xs: &Tensor<T>, drop_p: T) -> NnResult<Tensor<T>> {
    if drop_p < T::zero() || drop_p >= T::one() {
        Err(NnError::DropoutInvalid(drop_p.to_f64()))?;
    }

    let rand = Tensor::<T>::rand(T::zero(), T::one(), xs.shape())?;
    let scale = T::one() / (T::one() - drop_p);
    let mask = rand.ge(drop_p)?.cast()? * scale;
    Ok(xs * mask)
}

/// A simple lookup table that looks up embeddings in a fixed dictionary and size.
/// 
/// ## Arguments
/// 
/// * `weight` - Embedding weight / Lookup table
/// * `indexes` - Index tensor, can be all shape, it will be flatten
/// 
/// ## Returns
/// 
/// * A Tensor of shape: indexes.dims() + embedding_sizes
pub fn embedding<T: WithDType>(weight: &Tensor<T>, indexes: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
    let (_, embedding_size) = weight
        .dims2()
        .map_err(NnError::Core)
        .context("embedding weight should be dim2")?;

    let indexes = indexes.into();
    let mut final_dims = indexes.dims().to_vec();
    final_dims.push(embedding_size);

    let indexes = indexes.flatten_all()?;
    let values = weight.index_select(indexes, 0)?;
    let values = values.reshape(final_dims)?;
    Ok(values)
}