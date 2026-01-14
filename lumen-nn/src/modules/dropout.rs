use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::functional as F;

#[derive(Module)]
#[module(display = "display")]
pub struct Dropout<T: FloatDType> {
    #[module(skip)]
    drop_p: T,
}

impl<T: FloatDType> Dropout<T> {
    pub fn new(drop_p: T) -> Self {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor<T>, train: bool) -> lumen_core::Result<Tensor<T>> {
        if train {
            F::dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }

    fn display(&self) -> String {
        format!("drop_p={}", self.drop_p)
    }
}

