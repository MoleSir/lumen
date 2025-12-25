use lumen_core::{Dim, FloatDType, IntTensor, Tensor, D};

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use lumen_core::Tensor;
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]]).unwrap();
/// let a = lumen_nn::functional::softmax(&a, 1).unwrap();
/// ```
pub fn softmax<T: FloatDType, D: Dim>(xs: &Tensor<T>, dim: D) -> lumen_core::Result<Tensor<T>> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?; // (..., 1, ...)
    let diff = xs.broadcast_sub(&max)?;   // (..., D, ...)
    let num = diff.exp();                 // (..., D, ...)
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<T: FloatDType, D: Dim>(xs: &Tensor<T>, dim: D) -> lumen_core::Result<Tensor<T>> {
    let dim = dim.to_index(xs.shape(), "log_softmax")?;
    let max = xs.max_keepdim(dim)?; // (..., 1, ...)
    let diff = xs.broadcast_sub(&max)?;   // (..., D, ...)
    let num = diff.exp().sum_keepdim(dim)?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn silu<T: FloatDType>(xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    Ok(xs.silu())
}

pub fn swiglu<T: FloatDType>(xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    let xs = xs.chunk(2, D::Minus1)?;
    Ok(xs[0].silu() * &xs[1])
}

pub fn sigmoid<T: FloatDType>(xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    Ok(xs.sigmoid())
}

pub fn hard_sigmoid<T: FloatDType>(xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    ((xs + T::from_f64(3.0)) / T::from_f64(6.0)).clamp(T::zero(), T::one())
}

pub fn leaky_relu<T: FloatDType>(xs: &Tensor<T>, negative_slope: T) -> lumen_core::Result<Tensor<T>> {
    Ok(xs.leaky_relu(negative_slope))
}

pub fn dropout<T: FloatDType>(xs: &Tensor<T>, drop_p: f64) -> lumen_core::Result<Tensor<T>> {
    if !(0. ..1.).contains(&drop_p) {
        lumen_core::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }

    let rand = Tensor::<T>::rand(T::zero(), T::one(), xs.shape())?;
    let scale = T::from_f64(1.0 / (1.0 - drop_p));
    let drop_p = Tensor::new(scale)?.broadcast_as(xs.shape())?;
    let mask = rand.ge(drop_p)?.to_dtype() * scale;
    Ok(xs * mask)
}

#[allow(unused)]
pub fn null_loss<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
    unimplemented!("null_loss")
}

pub fn mse_loss<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    input.sub(target)?.sqr().mean_all()
}

#[allow(unused)]
pub fn cross_entropy<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    unimplemented!("cross_entropy")
}

