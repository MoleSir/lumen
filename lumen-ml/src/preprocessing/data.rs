use lumen_core::{FloatDType, Tensor, D};

pub struct StandardScaler<T> {
    pub eps: T,
}

impl<T: FloatDType> StandardScaler<T> {
    pub fn transform(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        let std = (var + self.eps).sqrt()?;

        let y = x
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?;

        Ok(y)
    }
}

pub struct MinMaxScaler<T> {
    pub eps: T,
}

impl<T: FloatDType> MinMaxScaler<T> {
    pub fn transform(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let min = x.min(D::Minus1)?;
        let max = x.max(D::Minus1)?;
        let delta = &max - &min;

        let y = x
            .broadcast_sub(&min)?
            .broadcast_div(&(delta + self.eps))?;

        Ok(y)
    }
}
