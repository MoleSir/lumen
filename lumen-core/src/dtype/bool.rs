use crate::{DynTensor, Tensor};
use super::{DType, NoAutograd, WithDType};

impl WithDType for bool {
    const DTYPE: DType = DType::Bool;
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

