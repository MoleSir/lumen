mod f32;
mod f64;
mod u32;
mod i32;
mod bool;
mod u8;

use crate::{grad::{AutogradInfo, AutogradMetaT, NoAutograd}, DynTensor, IntTensor, Result, Tensor};
use super::Storage;

pub trait WithDType:
    Sized
    + Copy
    + std::cmp::PartialOrd
    + std::cmp::PartialEq
    + std::fmt::Display
    + Boolean
    + 'static
    + Send
    + Sync
{
    const DTYPE: DType;
    type AutogradMeta: AutogradMetaT<Self>;
    fn from_dyn(tensor: &DynTensor) -> crate::Result<Tensor<Self>>;
    fn into_dyn(tensor: Tensor<Self>) -> DynTensor;
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,  // boolean
    U8,    // unsigned 8-bit
    U32,   // unsigned 32-bit
    I32,   // signed 32-bit
    F32,   // 32-bit float
    F64,   // 64-bit float
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::Bool => std::mem::size_of::<bool>(),
            DType::U8 => std::mem::size_of::<u8>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => std::mem::size_of::<f64>(),
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => write!(f, "boolean"),
            Self::U8   => write!(f, "uint8"),
            Self::I32 => write!(f, "int32"),
            Self::U32 => write!(f, "uint32"),
            Self::F32 => write!(f, "float32"),
            Self::F64 => write!(f, "float64"),
        }
    }
}

pub trait NumDType: 
    WithDType 
  + num_traits::Num    
  + num_traits::Bounded
  + rand_distr::uniform::SampleUniform
  + std::iter::Sum
  + std::iter::Product
  + std::ops::AddAssign
  + std::ops::SubAssign
  + std::ops::MulAssign
  + std::ops::DivAssign
  + std::ops::Add<Tensor<Self>, Output = Tensor<Self>> 
  + std::ops::Mul<Tensor<Self>, Output = Tensor<Self>>  
  + std::ops::Sub<Tensor<Self>, Output = Tensor<Self>>  
  + std::ops::Div<Tensor<Self>, Output = Tensor<Self>> 
  + for<'a> std::ops::Add<&'a Tensor<Self>, Output = Tensor<Self>> 
  + for<'a> std::ops::Mul<&'a Tensor<Self>, Output = Tensor<Self>> 
  + for<'a> std::ops::Sub<&'a Tensor<Self>, Output = Tensor<Self>> 
  + for<'a> std::ops::Div<&'a Tensor<Self>, Output = Tensor<Self>>  
{
    type Category: NumCategory;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;

    fn minimum(lhs: Self, rhs: Self) -> Self;
    fn maximum(lhs: Self, rhs: Self) -> Self;
    fn close(self, other: Self, rtol: f64, atol: f64) -> bool;

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>>;
}

pub trait IntDType: 
    NumDType<Category = IntCategory>
    + num_traits::Bounded 
    + num_traits::Pow<usize>
    + Ord
{
    fn is_true(self) -> bool {
        self != Self::zero()
    }

    fn to_inttensor(tensor: Tensor<Self>) -> IntTensor;
}

pub trait SignedIntDType : 
    IntDType 
  + num_traits::Signed 
{
    fn abs(self) -> Self;
    fn neg(self) -> Self;
}

pub trait UnsignedIntDType : 
    IntDType 
  + num_traits::Unsigned 
{
}

pub trait FloatDType: 
    NumDType<Category = FloatCategory, AutogradMeta = AutogradInfo<Self>>
    + num_traits::Float
    + rand_distr::num_traits::Float
{
    fn sqr(self) -> Self;
    fn gelu(self) -> Self;
    fn gelu_erf(self) -> Self;
    fn erf(self) -> Self;
    fn relu(self) -> Self;
    fn leaky_relu(self, negative_slope: Self) -> Self; 
    fn silu(self) -> Self;
    fn sigmoid(self) -> Self;

    fn two() -> Self;
    fn pi() -> Self;
    fn half() -> Self;

    fn min_value() -> Self;

    fn random_normal_vec(count: usize, mean: Self, std: Self) -> crate::Result<Vec<Self>>;
}

pub trait NumCategory {}
pub struct IntCategory {}
pub struct FloatCategory {}

impl NumCategory for IntCategory {}
impl NumCategory for FloatCategory {}

pub trait DTypeConvert<To: WithDType>: WithDType {
    fn convert(self) -> To;
}

macro_rules! impl_dtype_convert_from {
    ($from:ty, { $($to:ty),* }) => {
        $(
            impl DTypeConvert<$to> for $from {
                #[inline]
                fn convert(self) -> $to {
                    self as $to
                }
            }
        )*
    };
}

impl_dtype_convert_from!(u8,  { u8, i32, u32, f32, f64 });
impl_dtype_convert_from!(i32, { u8, i32, u32, f32, f64 });
impl_dtype_convert_from!(u32, { u8, i32, u32, f32, f64 });
impl_dtype_convert_from!(f32, { u8, i32, u32, f32, f64 });
impl_dtype_convert_from!(f64, { u8, i32, u32, f32, f64 });

impl<T: NumDType> DTypeConvert<T> for bool {
    fn convert(self) -> T {
        if self { T::one() } else { T::zero() }
    }
}

impl<T: NumDType> DTypeConvert<bool> for T {
    fn convert(self) -> bool {
        self == T::zero() 
    }
}

pub trait Boolean {
    fn true_value() -> Self;
    fn false_value() -> Self;
}

macro_rules! impl_boolean_for_int {
    ($($t:ty),*) => {
        $(
            impl Boolean for $t {
                fn false_value() -> Self {
                    0
                }
            
                fn true_value() -> Self {
                    1
                }
            }
        )*
    };
}

macro_rules! impl_boolean_for_float {
    ($($t:ty),*) => {
        $(
            impl Boolean for $t {
                fn false_value() -> Self {
                    0.
                }
            
                fn true_value() -> Self {
                    1.
                }
            }
        )*
    };
}

impl_boolean_for_int!(u8, i32, u32);
impl_boolean_for_float!(f32, f64);

impl Boolean for bool {
    fn false_value() -> Self {
        false
    }

    fn true_value() -> Self {
        true
    }
}
