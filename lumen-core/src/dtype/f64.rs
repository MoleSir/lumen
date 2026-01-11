use approx::relative_eq;
use rand::rng;
use rand_distr::Distribution;
use crate::{DynTensor, Result, Storage, Tensor};
use super::{AutogradInfo, DType, FloatCategory, FloatDType, NumDType, WithDType};

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;
    type AutogradMeta = AutogradInfo<f64>;

    fn from_dyn(tensor: &DynTensor) -> crate::Result<Tensor<Self>> {
        if let DynTensor::F64(t) = tensor {
            Ok(t.clone())
        } else {
            Err(crate::Error::UnexpectedDType { msg: "convert from dyn tensor", expected: Self::DTYPE, got: tensor.dtype() })
        }
    }
}

impl NumDType for f64 {
    type Category = FloatCategory;
    
    fn from_f64(v: f64) -> Self {
        v as f64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn minimum(lhs: Self, rhs: Self) -> Self {
        if lhs > rhs { rhs } else { lhs }
    }

    fn maximum(lhs: Self, rhs: Self) -> Self {
        if lhs < rhs { rhs } else { lhs }
    }

    fn close(self, other: Self, rtol: f64, atol: f64) -> bool {
        relative_eq!(self, other, epsilon = atol, max_relative = rtol)
    }

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let mut vec = vec![];
        let mut v = start;
        while v < end {
            vec.push(v);
            v += 1.0;
        }
        Ok(Storage::new(vec))
    }
}

impl FloatDType for f64 {
    fn min_value() -> Self {    
        f64::MIN
    }
    
    #[inline]
    fn sqr(self) -> Self {
        self * self
    }

    #[inline]
    fn two() -> Self {
        2.
    }

    #[inline]
    fn pi() -> Self {
        std::f64::consts::PI
    }

    #[inline]
    fn half() -> Self {
        0.5
    }

    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu(self) -> Self {
        const SQRT_2_OVER_PI: f64 = 0.79788456; // sqrt(2/pi)
        const COEF: f64 = 0.044715;
        let x_cubed = self * self * self;
        let inner = SQRT_2_OVER_PI * (self + COEF * x_cubed);
        
        0.5 * self * (1.0 + inner.tanh())
    }

    /// 0.5 * x * (1 + erf(x / sqrt(2)))
    fn gelu_erf(self) -> Self {
        const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2; // 0.70710678
        0.5 * self * (1.0 + (self * FRAC_1_SQRT_2).erf())
    }

    #[inline]
    fn erf(self) -> Self {
        libm::erf(self)
    }

    /// ReLU
    #[inline]
    fn relu(self) -> Self {
        self.max(0.0)
    }

    fn leaky_relu(self, negative_slope: Self) -> Self {
        if self > 0.0 {
            self
        } else {
            self * negative_slope
        }
    }

    /// SiLU (Swish)
    fn silu(self) -> Self {
        self / (1.0 + (-self).exp())
    }

    fn sigmoid(self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    fn random_normal_vec(count: usize, mean: Self, std: Self) -> crate::Result<Vec<Self>> {
        let normal = rand_distr::Normal::new(mean, std).map_err(|e| crate::Error::Rand(e.to_string()))?;
        let mut rng = rng();
        let v: Vec<Self> = (0..count)
            .map(|_| normal.sample(&mut rng))
            .collect();
        Ok(v)
    }
}
