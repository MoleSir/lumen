use lumen_core::{FloatDType, Tensor, D};
use crate::NnResult;

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