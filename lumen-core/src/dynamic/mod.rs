mod float;
mod integer;
pub use float::*;
pub use integer::*;
use crate::{DType, Shape, Tensor, WithDType};
use paste::paste;
use half::bf16;

#[derive(Clone)]
pub enum DynTensor {
    Bool(Tensor<bool>),
    Bf16(Tensor<bf16>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    U32(Tensor<u32>),
    U8(Tensor<u8>),
}

macro_rules! method_for_all_dtype {
    ($self:ident, $t:ident, $expr:expr) => {
        match &$self {
            DynTensor::Bool($t) => $expr,
            DynTensor::Bf16($t) => $expr,
            DynTensor::F32($t) => $expr,
            DynTensor::F64($t) => $expr,
            DynTensor::U32($t) => $expr,
            DynTensor::I32($t) => $expr,
            DynTensor::U8($t) => $expr,
        }
    };
}

macro_rules! method_for_all_dtype_with_dyn {
    ($self:ident, $t:ident, $expr:expr) => {
        match &$self {
            DynTensor::Bool($t) => $expr.map(DynTensor::Bool),
            DynTensor::Bf16($t) => $expr.map(DynTensor::Bf16),
            DynTensor::F32($t) => $expr.map(DynTensor::F32),
            DynTensor::F64($t) => $expr.map(DynTensor::F64),
            DynTensor::U32($t) => $expr.map(DynTensor::U32),
            DynTensor::I32($t) => $expr.map(DynTensor::I32),
            DynTensor::U8($t) => $expr.map(DynTensor::U8),
        }
    };
}

impl DynTensor {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Bf16(_) => DType::Bf16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
        }
    }

    pub fn shape(&self) -> &Shape {
        method_for_all_dtype!(self, t, t.shape())
    }

    pub fn transpose_last(&self) -> crate::Result<Self> {
        method_for_all_dtype_with_dyn!(self, t, t.transpose_last())
    }

    pub fn contiguous(&self) -> crate::Result<Self> {
        method_for_all_dtype_with_dyn!(self, t, t.contiguous())
    }

    pub fn as_tensor<T: WithDType>(&self) -> crate::Result<Tensor<T>> {
        T::from_dyn(self)
    }
}

macro_rules! impl_convert_with_type {
    // Rule 1: Allow exact match and specific castable variants
    ($variant:ident, $inner:ty, [$($cast_variant:ident),*]) => {
        paste! {
            impl DynTensor {
                pub fn [< is_ $inner >](&self) -> bool {
                    matches!(self, Self::$variant(_))
                }

                pub fn [< as_ $inner >](&self) -> Option<Tensor<$inner>> {
                    match self {
                        Self::$variant(t) => Some(t.clone()),
                        // Note: `as_type` usually implies exact/cheap match. 
                        // If you want `.as_f32()` to do expensive casting, you can add it here, 
                        // but it's usually better practice to keep `as_` exact and rely on `TryFrom`/`cast` for casting.
                        _ => None,
                    }
                }
            }
    
            impl From<Tensor<$inner>> for DynTensor {
                fn from(t: Tensor<$inner>) -> Self {
                    DynTensor::$variant(t)
                }
            }

            impl From<&Tensor<$inner>> for DynTensor {
                fn from(t: &Tensor<$inner>) -> Self {
                    DynTensor::$variant(t.clone())
                }
            }
    
            impl TryFrom<DynTensor> for Tensor<$inner> {
                type Error = crate::Error;
    
                #[allow(unreachable_patterns)]
                fn try_from(value: DynTensor) -> Result<Self, Self::Error> {
                    match value {
                        // 1. Exact Match
                        DynTensor::$variant(t) => Ok(t),
                        
                        // 2. Castable Matches
                        // This expands to: `DynTensor::F64(t) => t.cast::<f32>(),` etc.
                        $(
                            DynTensor::$cast_variant(t) => t.cast::<$inner>(),
                        )*
                        
                        // 3. Fallback Error
                        // Notice we bind to `other` so we can call `.dtype()` without moving `value`
                        other => Err(crate::Error::UnexpectedDType { 
                            msg: "failed to cast in try_from", 
                            expected: DType::$variant, 
                            got: other.dtype() 
                        }),
                    }
                }
            }
        }
    };
    
    // Rule 2: Fallback for types that shouldn't accept casting from anything else (Empty array)
    ($variant:ident, $inner:ty) => {
        impl_convert_with_type!($variant, $inner, []);
    };
}

impl_convert_with_type!(Bf16, bf16, [F64, F32]);
impl_convert_with_type!(F32, f32, [F64, Bf16]);
impl_convert_with_type!(F64, f64, [F32, Bf16]);

// Types with no implicit casting allowed from DynTensor just use the standard call
impl_convert_with_type!(Bool, bool, [U8, U32, I32, Bf16, F32, F64]);
impl_convert_with_type!(I32, i32, [U8, U32, I32, Bf16, F32, F64]);
impl_convert_with_type!(U32, u32, [U8, I32, I32, Bf16, F32, F64]);
impl_convert_with_type!(U8, u8, [I32, U32, I32, Bf16, F32, F64]);
