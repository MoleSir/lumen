use crate::{DynTensor, Tensor};
use super::{DType, NoAutograd, WithDType};

impl WithDType for bool {
    const DTYPE: DType = DType::Bool;
    type AutogradMeta = NoAutograd;

    fn from_dyn(tensor: &DynTensor) -> crate::Result<Tensor<Self>> {
        if let DynTensor::Bool(t) = tensor {
            Ok(t.clone())
        } else {
            Err(crate::Error::UnexpectedDType { msg: "convert from dyn tensor", expected: Self::DTYPE, got: tensor.dtype() })
        }
    }
}

