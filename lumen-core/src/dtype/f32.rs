use approx::relative_eq;
use libm;
use crate::{Result, Storage};
use super::{AutogradInfo, DType, FloatCategory, FloatDType, NumDType, WithDType};

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
    type AutogradMeta = AutogradInfo<f32>;
}

impl NumDType for f32 {
    type Category = FloatCategory;

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f32
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
        relative_eq!(self, other, epsilon = atol as f32, max_relative = rtol as f32)
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

impl FloatDType for f32 {
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
        std::f32::consts::PI
    }

    #[inline]
    fn half() -> Self {
        0.5
    }

    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu(self) -> Self {
        const SQRT_2_OVER_PI: f32 = 0.79788456; // sqrt(2/pi)
        const COEF: f32 = 0.044715;
        let x_cubed = self * self * self;
        let inner = SQRT_2_OVER_PI * (self + COEF * x_cubed);
        
        0.5 * self * (1.0 + inner.tanh())
    }

    /// 0.5 * x * (1 + erf(x / sqrt(2)))
    fn gelu_erf(self) -> Self {
        const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 0.70710678
        0.5 * self * (1.0 + (self * FRAC_1_SQRT_2).erf())
    }

    #[inline]
    fn erf(self) -> Self {
        libm::erff(self)
    }

    /// ReLU
    #[inline]
    fn relu(self) -> Self {
        self.max(0.0)
    }

    /// SiLU (Swish)
    fn silu(self) -> Self {
        self / (1.0 + (-self).exp())
    }
}
