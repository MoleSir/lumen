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
    //  log(sum(exp(x - max)))
    let log_sum_exp = diff.exp().sum_keepdim(dim)?.ln();
    // (x - max) - log_sum_exp
    diff.broadcast_sub(&log_sum_exp)
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

pub fn nll_loss<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
    let target = target.into();
    let gathered = input.gather(target, 1)?;
    let neg_loss = gathered.neg();
    neg_loss.mean_all()
}

pub fn mse_loss<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    input.sub(target)?.sqr().mean_all()
}

pub fn cross_entropy<T: FloatDType>(input: &Tensor<T>, target: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
    // Input Shape: [Batch, Classes] (logits)
    // Target Shape: [Batch, Classes] (probabilities, 0.0~1.0)
    let dim = 1; 
    let log_probs = log_softmax(input, dim)?;
    let weighted_log_probs = target.broadcast_mul(&log_probs)?;
    let sum_class = weighted_log_probs.sum_keepdim(dim)?;
    let loss = sum_class.neg();
    loss.mean_all()
}

pub fn cross_entropy_indices<T: FloatDType>(input: &Tensor<T>, target: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
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