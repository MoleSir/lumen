use crate::{DynTensor, IntTensor, Result, Storage, Tensor};

use super::{DType, IntCategory, IntDType, NoAutograd, NumDType, UnsignedIntDType, WithDType};

impl WithDType for u8 {
    const DTYPE: DType = DType::U8;
    const ZERO: Self = 0;
    const ONE: Self = 1;
    type AutogradMeta = NoAutograd;

    #[inline]
    fn from_dyn(tensor: &DynTensor) -> crate::Result<Tensor<Self>> {
        <Tensor<Self> as TryFrom::<DynTensor>>::try_from(tensor.clone())
    }

    #[inline]
    fn into_dyn(tensor: Tensor<Self>) -> DynTensor {
        tensor.into()
    }
}

impl NumDType for u8 {

    const MAX_VALUE: Self = u8::MAX;
    const MIN_VALUE: Self = u8::MIN;

    type Category = IntCategory;

    fn from_f64(v: f64) -> Self {
        v as u8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as u8
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

    fn close(self, other: Self, _rtol: f64, _atol: f64) -> bool {
        self == other
    }

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let vec: Vec<_> = (start..end).collect();
        Ok(Storage::new(vec))
    }
}

impl IntDType for u8 {
    fn to_inttensor(tensor: Tensor<Self>) -> IntTensor {
        IntTensor::U8(tensor)
    }
}

impl UnsignedIntDType for u8 {}
