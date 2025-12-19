use super::{DType, NoAutograd, WithDType};

impl WithDType for bool {
    const DTYPE: DType = DType::Bool;
    type AutogradMeta = NoAutograd;
}

