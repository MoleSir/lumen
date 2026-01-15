use lumen_core::{Dim, FloatDType, IntTensor, NumDType, Tensor, D};
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
    let num = diff.exp();                 // (..., D, ...)
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
    let log_sum_exp = diff.exp().sum_keepdim(dim)?.ln();
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
    let mask = rand.ge(drop_p)?.to_dtype() * scale;
    Ok(xs * mask)
}

// ============================================================================= //
//                         Activate 
// ============================================================================= //

/// Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
///
/// The SiLU activation function is also known as the swish function: $SiLU(x) = x * \sigma(x)$.
pub fn silu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.silu())
}

/// Applies the SwiGLU activation.
///
/// Expects the input tensor to have an even size in the last dimension.
/// It splits the input into two halves (A and B) along the last dimension,
/// and computes $SiLU(A) \otimes B$.
///
/// ## Arguments
///
/// * `xs` - Input tensor. Shape `(..., 2 * d)`.
///
/// ## Returns
///
/// * Output tensor. Shape `(..., d)`.
pub fn swiglu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    let xs = xs.chunk(2, D::Minus1)?;
    Ok(xs[0].silu() * &xs[1])
}

/// Applies the Sigmoid function, element-wise.
///
/// $\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$
pub fn sigmoid<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.sigmoid())
}

/// Applies the Rectified Linear Unit (ReLU) function, element-wise.
///
/// $\text{ReLU}(x) = \max(0, x)$
pub fn relu<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    Ok(xs.relu())
}

/// Applies the Hard Sigmoid function, element-wise.
///
/// Defined as $\frac{\text{ReLU6}(x + 3)}{6}$.
/// Result is clamped between 0 and 1.
pub fn hard_sigmoid<T: FloatDType>(xs: &Tensor<T>) -> NnResult<Tensor<T>> {
    let out = ((xs + T::from_f64(3.0)) / T::from_f64(6.0)).clamp(T::zero(), T::one())?;
    Ok(out)
}

/// Applies the Leaky ReLU function, element-wise.
///
/// $\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)$
///
/// ## Arguments
///
/// * `xs` - Input tensor.
/// * `negative_slope` - Controls the angle of the negative slope.
pub fn leaky_relu<T: FloatDType>(xs: &Tensor<T>, negative_slope: T) -> NnResult<Tensor<T>> {
    Ok(xs.leaky_relu(negative_slope))
}

// ============================================================================= //
//                         Loss 
// ============================================================================= //

/// Computes the Negative Log Likelihood loss.
///
/// Expects the input to contain log-probabilities (e.g., from `log_softmax`).
///
/// ## Arguments
///
/// * `input` - Log-probabilities of shape `(batch, num_classes)`.
/// * `target` - Class indices of shape `(batch, 1)`.
///
/// ## Returns
///
/// * Scalar tensor representing the mean loss.
pub fn nll_loss<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> NnResult<Tensor<T>> {
    let target = target.into();
    let gathered = input.gather(target, 1)?;
    let neg_loss = gathered.neg();
    let out = neg_loss.mean_all()?;
    Ok(out)
}

/// Computes the Mean Squared Error loss.
///
/// Measures the element-wise mean squared error.
///
/// ## Arguments
///
/// * `input` - Input tensor.
/// * `target` - Target tensor.
///
/// ## Returns
///
/// * Scalar tensor representing the mean loss.
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

// ============================================================================= //
//                         Normalization 
// ============================================================================= //

/// Applies Layer Normalization over a mini-batch of inputs.
///
/// $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
///
/// The mean and standard-deviation are calculated over the last dimension.
///
/// ## Arguments
///
/// * `input` - Input tensor of shape `(..., normalized_shape)`.
/// * `weight` - Optional scale tensor ($\gamma$) of shape `(normalized_shape)`.
/// * `bias` - Optional shift tensor ($\beta$) of shape `(normalized_shape)`.
/// * `eps` - A value added to the denominator for numerical stability.
pub fn layer_norm<T: FloatDType>(
    input: &Tensor<T>,
    weight: Option<&Tensor<T>>,
    bias: Option<&Tensor<T>>,
    eps: T,
) -> NnResult<Tensor<T>> {
    // 1. Calculate Mean and Variance along the last dimension
    let mean = input.mean_keepdim(D::Minus1)?;
    let var = input.var_keepdim(D::Minus1)?;

    // 2. Normalize: (x - mean) / sqrt(var + eps)
    let std = (var + eps).sqrt();
    let input_normalized = input
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?;

    // 3. Apply Affine Transform: x * weight + bias
    let out = match (weight, bias) {
        (Some(w), Some(b)) => input_normalized.broadcast_mul(w)?.broadcast_add(b)?,
        (Some(w), None) => input_normalized.broadcast_mul(w)?,
        (None, Some(b)) => input_normalized.broadcast_add(b)?,
        (None, None) => input_normalized,
    };

    Ok(out)
}

/// Applies Root Mean Square Normalization.
///
/// $\text{RMS}(x) = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \cdot \gamma$
///
/// Commonly used in LLMs (e.g., LLaMA). It normalizes inputs based on the root mean square
/// without centering the mean.
///
/// ## Arguments
///
/// * `input` - Input tensor.
/// * `weight` - Scale tensor ($\gamma$).
/// * `eps` - A value added to the denominator for numerical stability.
pub fn rms_norm<T: FloatDType>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    eps: T,
) -> NnResult<Tensor<T>> {
    // 1. Calculate Mean of Squares: Mean(x^2)
    let variance = input.sqr().mean_keepdim(D::Minus1)?;

    // 2. Calculate RMS: sqrt(variance + eps)
    let rms = (variance + eps).sqrt();

    // 3. Normalize: x / rms
    let input_normalized = input.broadcast_div(&rms)?;

    // 4. Scale: x * weight
    let out = input_normalized.broadcast_mul(weight)?;

    Ok(out)
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