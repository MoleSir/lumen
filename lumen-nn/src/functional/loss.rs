use lumen_core::{FloatDType, IntTensor, Tensor};
use crate::{NnError, NnResult};
use super::log_softmax;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossReduction {
    None,
    #[default]
    Mean,
    Sum,
}

impl LossReduction {
    pub(crate) fn loss<T: FloatDType>(&self, loss: Tensor<T>) -> NnResult<Tensor<T>> {
        let loss = match self {
            Self::None => loss,
            Self::Mean => loss.mean_all().map_err(NnError::Core)?,
            Self::Sum => loss.sum_all().map_err(NnError::Core)?,
        };
        Ok(loss)
    }

    pub(crate) fn to_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Mean => "mean",
            Self::Sum => "sum",
        }
    }
}

/// Computes the L1 loss (Mean Absolute Error).
///
/// L1 Loss = |input - target|
///
/// ## Arguments
///
/// * `input` - Input tensor. Shape: `(batch, ...)`
/// * `target` - Target tensor. Shape: `(batch, ...)` (Must match input shape)
/// * `reduction` - Specifies the reduction to apply to the output: `None` | `Mean` | `Sum`.
pub fn l1_loss<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<T>, 
    reduction: LossReduction
) -> NnResult<Tensor<T>> {
    let loss = input.sub(target)?.abs();
    reduction.loss(loss)
}

/// Computes the Mean Squared Error (MSE) loss.
///
/// MSE Loss = (input - target)^2
///
/// ## Arguments
///
/// * `input` - Input tensor. Shape: `(batch, ...)`
/// * `target` - Target tensor. Shape: `(batch, ...)` (Must match input shape)
/// * `reduction` - Specifies the reduction to apply to the output: `None` | `Mean` | `Sum`.
pub fn mse_loss<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<T>,
    reduction: LossReduction
) -> NnResult<Tensor<T>> {
    let loss = input.sub(target)?.sqr();
    reduction.loss(loss)
}

/// Computes the Negative Log Likelihood (NLL) loss.
///
/// This function expects the input to contain **log-probabilities** (e.g., output of `log_softmax`).
///
/// ## Arguments
///
/// * `input` - Log-probabilities.
///   **Shape:** `(batch, num_classes)`
///
/// * `target` - Class indices.
///   **Shape:** `(batch, 1)` or `(batch)` depending on gathering implementation.
///   Values must be in range `[0, num_classes - 1]`.
///
/// * `reduction` - Specifies the reduction to apply to the output: `None` | `Mean` | `Sum`.
///
/// ## Returns
///
/// * If `reduction` is `None`: Tensor of shape `(batch, 1)`.
/// * If `reduction` is `Mean` or `Sum`: Scalar Tensor.
pub fn nll_loss<T: FloatDType>(
    input: &Tensor<T>, 
    target: impl Into<IntTensor>,
    reduction: LossReduction,
) -> NnResult<Tensor<T>> {
    let target = target.into();
    // Gather logic: select the value at the target index for each sample
    let gathered = input.gather(target, 1)?;
    let neg_loss = gathered.neg();
    reduction.loss(neg_loss)
}

/// Computes the Cross Entropy Loss between input logits and target **probabilities**.
///
/// This function applies `log_softmax` on the input implicitly.
/// It is useful when training with "soft" labels (e.g., label smoothing, MixUp).
///
/// ## Arguments
///
/// * `input` - The input tensor containing unnormalized scores (logits).
///   **Shape:** `(batch, num_classes)`
///
/// * `target` - The target tensor containing probabilities (0.0 to 1.0).
///   **Shape:** `(batch, num_classes)` (Must match `input` shape).
///
/// * `reduction` - Specifies the reduction to apply to the output: `None` | `Mean` | `Sum`.
///
/// ## Returns
///
/// * If `reduction` is `None`: Tensor of shape `(batch, 1)`.
/// * If `reduction` is `Mean` or `Sum`: Scalar Tensor.
pub fn soft_cross_entropy<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<T>,
    reduction: LossReduction,
) -> NnResult<Tensor<T>> {
    let dim = 1; 
    let log_probs = log_softmax(input, dim)?;
    
    // H(p, q) = - sum(p(x) * log(q(x)))
    let weighted_log_probs = target.broadcast_mul(&log_probs)?;
    // Sum across classes -> shape: (batch, 1)
    let sum_class = weighted_log_probs.sum_keepdim(dim)?;
    let loss = sum_class.neg();
    
    reduction.loss(loss)
}

/// Computes the Cross Entropy Loss using integer **class indices** as targets.
///
/// This is the standard loss function for multi-class classification.
/// It combines `log_softmax` and `nll_loss` for numerical stability.
///
/// ## Arguments
///
/// * `input` - The input tensor containing unnormalized scores (logits).
///   **Shape:** `(batch, num_classes)`
///
/// * `target` - The target tensor containing class indices.
///   **Shape:** `(batch, 1)`
///   Values must be integers in the range `[0, num_classes - 1]`.
///
/// * `reduction` - Specifies the reduction to apply to the output: `None` | `Mean` | `Sum`.
///
/// ## Returns
///
/// * If `reduction` is `None`: Tensor of shape `(batch, 1)`.
/// * If `reduction` is `Mean` or `Sum`: Scalar Tensor.
pub fn cross_entropy<T: FloatDType>(
    input: &Tensor<T>, 
    target: impl Into<IntTensor>,
    reduction: LossReduction,
) -> NnResult<Tensor<T>> {
    // Input: [batch, classes] (logits)
    // Target: [batch, 1] (indices)
    // 1. Compute Log Softmax
    let log_probs = log_softmax(input, 1)?;
    // 2. Compute NLL Loss
    nll_loss(&log_probs, target, reduction)
}