use core::f32;
use approx::relative_eq;
use half::bf16;
use libm;
use num_traits::{real::Real, FromPrimitive, ToPrimitive};
use rand::rng;
use rand_distr::Distribution;
use crate::{DynTensor, Result, Storage, Tensor};
use super::{AutogradInfo, DType, DTypeConvert, FloatCategory, FloatDType, NumDType, WithDType};

impl WithDType for half::bf16 {
    const DTYPE: DType = DType::F32;
    const ZERO: Self = half::bf16::ZERO;
    const ONE: Self = half::bf16::ONE;
    type AutogradMeta = AutogradInfo<half::bf16>;

    #[inline]
    fn from_dyn(tensor: &DynTensor) -> crate::Result<Tensor<Self>> {
        <Tensor<Self> as TryFrom::<DynTensor>>::try_from(tensor.clone())
    }

    #[inline]
    fn into_dyn(tensor: Tensor<Self>) -> DynTensor {
        tensor.into()
    }
}

impl NumDType for half::bf16 {
    type Category = FloatCategory;

    const MAX_VALUE: Self = half::bf16::MAX;
    const MIN_VALUE: Self = half::bf16::MIN;

    fn from_f64(v: f64) -> Self {
        half::bf16::from_f64(v)
    }

    fn to_f64(self) -> f64 {
        half::bf16::to_f64(self)
    }

    fn from_usize(v: usize) -> Self {
        half::bf16::from_u64(v as u64).expect("")
    }

    fn to_usize(self) -> usize {
        half::bf16::to_u64(&self).expect("") as usize
    }

    fn minimum(lhs: Self, rhs: Self) -> Self {
        if lhs > rhs { rhs } else { lhs }
    }

    fn maximum(lhs: Self, rhs: Self) -> Self {
        if lhs < rhs { rhs } else { lhs }
    }

    fn close(self, other: Self, rtol: f64, atol: f64) -> bool {
        let a = Self::to_f32(self);
        let b = Self::to_f32(other);
        relative_eq!(a, b, epsilon = atol as f32, max_relative = rtol as f32)
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let mut vec = vec![];
        let mut v = start;
        while v < end {
            vec.push(v);
            v += half::bf16::ONE;
        }
        Ok(Storage::new(vec))
    }
}

impl FloatDType for half::bf16 {
    #[inline]
    fn sqr(self) -> Self {
        self * self
    }

    #[inline]
    fn two() -> Self {
        half::bf16::from_f64_const(2.0)
    }

    #[inline]
    fn pi() -> Self {
        half::bf16::from_f64_const(std::f64::consts::PI)
    }

    #[inline]
    fn half() -> Self {
        half::bf16::from_f64_const(0.5)
    }

    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu(self) -> Self {
        const SQRT_2_OVER_PI: bf16 = bf16::from_f32_const(0.79788456); // sqrt(2/pi)
        const COEF: bf16 = bf16::from_f32_const(0.044715);
        let x_cubed = self * self * self;
        let inner = SQRT_2_OVER_PI * (self + COEF * x_cubed);
        
        Self::half() * self * (bf16::ONE + inner.tanh())
    }

    /// 0.5 * x * (1 + erf(x / sqrt(2)))
    fn gelu_erf(self) -> Self {
        const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 0.70710678
        Self::half() * self * (bf16::ONE + (self * bf16::from_f32_const(FRAC_1_SQRT_2)).erf())
    }

    #[inline]
    fn erf(self) -> Self {
        let v = libm::erff(bf16::to_f32(self));
        bf16::from_f32(v)
    }

    /// ReLU
    #[inline]
    fn relu(self) -> Self {
        self.max(bf16::ZERO)
    }

    fn leaky_relu(self, negative_slope: Self) -> Self {
        if self > bf16::ZERO {
            self
        } else {
            self * negative_slope
        }
    }

    /// SiLU (Swish)
    fn silu(self) -> Self {
        self / (bf16::ZERO + (-self).exp())
    }

    fn sigmoid(self) -> Self {
        bf16::ONE / (bf16::ONE + (-self).exp())
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

impl DTypeConvert<f32> for bf16 {
    #[inline]
    fn convert(self) -> f32 {
        bf16::to_f32(self)
    }
}

impl DTypeConvert<bf16> for f32 {
    #[inline]
    fn convert(self) -> bf16 {
        bf16::from_f32(self)
    }
}

impl DTypeConvert<f64> for bf16 {
    #[inline]
    fn convert(self) -> f64 {
        bf16::to_f64(self)
    }
}

impl DTypeConvert<bf16> for f64 {
    #[inline]
    fn convert(self) -> bf16 {
        bf16::from_f64(self)
    }
}

impl DTypeConvert<u32> for bf16 {
    #[inline]
    fn convert(self) -> u32 {
        bf16::to_u32(&self).expect("to u32")
    }
}

impl DTypeConvert<bf16> for u32 {
    #[inline]
    fn convert(self) -> bf16 {
        bf16::from_u32(self).expect("from u32")
    }
}

impl DTypeConvert<i32> for bf16 {
    #[inline]
    fn convert(self) -> i32 {
        bf16::to_i32(&self).expect("to i32")
    }
}

impl DTypeConvert<bf16> for i32 {
    #[inline]
    fn convert(self) -> bf16 {
        bf16::from_i32(self).expect("from i32")
    }
}

impl DTypeConvert<u8> for bf16 {
    #[inline]
    fn convert(self) -> u8 {
        bf16::to_u8(&self).expect("to u8")
    }
}

impl DTypeConvert<bf16> for u8 {
    #[inline]
    fn convert(self) -> bf16 {
        bf16::from_u8(self).expect("from u8")
    }
}
