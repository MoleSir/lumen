mod float;
mod integer;
pub use float::*;
pub use integer::*;
use crate::{DType, Shape, Tensor, WithDType};
use paste::paste;

#[derive(Clone)]
pub enum DynTensor {
    Bool(Tensor<bool>),
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I32(Tensor<i32>),
    U32(Tensor<u32>),
    U8(Tensor<u8>),
}

impl DynTensor {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            Self::I32(_) => DType::I32,
            Self::U32(_) => DType::U32,
        }
    }

    pub fn shape(&self) -> &Shape {
        match self {
            Self::Bool(t) => t.shape(),
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::U8(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::U32(t) => t.shape(),
        }
    }

    pub fn as_tensor<T: WithDType>(&self) -> crate::Result<Tensor<T>> {
        T::from_dyn(self)
    }
}

macro_rules! impl_convert_with_type {
    ($variant:ident, $inner:ty) => {
        paste! {
            impl DynTensor {
                pub fn [< is_ $inner >](&self) -> bool {
                    match self {
                        Self::$variant(_) => true,
                        _ => false,
                    }
                }

                pub fn [< as_ $inner >](&self) -> Option<Tensor<$inner>> {
                    match self {
                        Self::$variant(t) => Some(t.clone()),
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
    
                fn try_from(value: DynTensor) -> Result<Self, Self::Error> {
                    match value {
                        DynTensor::$variant(t) => Ok(t),
                        _ => Err(crate::Error::UnexpectedDType { msg: "in dyn as", expected: DType::$variant, got: value.dtype() }),
                    }
                }
            }
        }
    };
}

impl_convert_with_type!(Bool, bool);
impl_convert_with_type!(F32, f32);
impl_convert_with_type!(F64, f64);
impl_convert_with_type!(I32, i32);
impl_convert_with_type!(U32, u32);
impl_convert_with_type!(U8, u8);

