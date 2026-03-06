use lumen_core::{FloatDType, Tensor, D};
use crate::NnResult;

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
    let std = (var + eps).sqrt()?;
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
    let variance = input.sqr()?.mean_keepdim(D::Minus1)?;

    // 2. Calculate RMS: sqrt(variance + eps)
    let rms = (variance + eps).sqrt()?;

    // 3. Normalize: x / rms
    let input_normalized = input.broadcast_div(&rms)?;

    // 4. Scale: x * weight
    let out = input_normalized.broadcast_mul(weight)?;

    Ok(out)
}