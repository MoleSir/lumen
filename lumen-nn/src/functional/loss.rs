use lumen_core::{FloatDType, IntTensor, Tensor};
use crate::{NnError, NnResult};
use super::log_softmax;

#[derive(Debug, Clone, Copy)]
pub enum LossReduction {
    Mean,
    Sum,
}

pub(crate) fn reduction_loss<T: FloatDType>(loss: Tensor<T>, reduction: Option<LossReduction>) -> NnResult<Tensor<T>> {
    let loss = match reduction {
        None => loss,
        Some(LossReduction::Mean) => loss.mean_all().map_err(NnError::Core)?,
        Some(LossReduction::Sum) => loss.mean_all().map_err(NnError::Core)?,
    };
    Ok(loss)
}

pub(crate) fn reduction_display(reduction: Option<LossReduction>) -> &'static str {
    match reduction {
        None => "none",
        Some(LossReduction::Mean) => "mean",
        Some(LossReduction::Sum) => "sum",
    }
}

pub fn l1_loss<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<T>, 
    reduction: Option<LossReduction>
) -> NnResult<Tensor<T>> {
    let loss = input.sub(target)?.abs();
    reduction_loss(loss, reduction)
}

/// Computes the Mean Squared Error loss.
///
/// ## Arguments
///
/// * `input` - Input tensor.
/// * `target` - Target tensor.
/// * `reduction` - None / Mean / Sum
pub fn mse_loss<T: FloatDType>(
    input: &Tensor<T>, 
    target: &Tensor<T>,
    reduction: Option<LossReduction>
) -> NnResult<Tensor<T>> {
    let loss = input.sub(target)?.sqr();
    reduction_loss(loss, reduction)
}

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