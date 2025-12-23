use lumen_core::{FloatDType, NumDType, Tensor};
use crate::functional as F;
use super::Module;

pub struct Dropout {
    drop_p: f64,
}

impl Dropout {
    pub fn new(drop_p: f64) -> Dropout {
        Self { drop_p }
    }

    pub fn forward<T: FloatDType>(&self, xs: &Tensor<T>, train: bool) -> lumen_core::Result<Tensor<T>> {
        if train {
            F::dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl<T: NumDType> Module<T> for Dropout {
}