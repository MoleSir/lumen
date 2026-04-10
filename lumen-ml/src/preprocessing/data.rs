use lumen_core::{FloatDType, Tensor, D};
use crate::pipeline::{TransformFit, TransformModel};

pub struct StandardScaler<T> {
    pub eps: T,
}

impl<T: FloatDType> Default for StandardScaler<T> {
    fn default() -> Self {
        Self { eps: T::epsilon() }
    }
}

pub struct StandardScalerModel<T: FloatDType> {
    pub mean: Tensor<T>,
    pub std: Tensor<T>,
}

impl<T: FloatDType> TransformFit for StandardScaler<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    type Model = StandardScalerModel<T>;

    fn fit(&self, x: &Tensor<T>) -> crate::error::MlResult<Self::Model> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;
        let std = (var + self.eps).sqrt()?;

        Ok(StandardScalerModel { mean, std })
    }
}

impl<T: FloatDType> TransformModel for StandardScalerModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn transform(&self, x: &Tensor<T>) -> crate::error::MlResult<Self::Output> {
        let y = x
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?;

        Ok(y)
    }
}

pub struct MinMaxScaler<T> {
    pub eps: T,
}

impl<T: FloatDType> Default for MinMaxScaler<T> {
    fn default() -> Self {
        Self { eps: T::epsilon() }
    }
}

pub struct MinMaxScalerModel<T: FloatDType> {
    pub min: Tensor<T>,
    pub delta: Tensor<T>,
}

impl<T: FloatDType> TransformFit for MinMaxScaler<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;
    type Model = MinMaxScalerModel<T>; 

    fn fit(&self, x: &Tensor<T>) -> crate::error::MlResult<Self::Model> {
        let min = x.min_keepdim(D::Minus1)?;
        let max = x.max_keepdim(D::Minus1)?;
        let delta = (&max - &min) + self.eps;
        Ok(MinMaxScalerModel { min, delta })
    }
}

impl<T: FloatDType> TransformModel for MinMaxScalerModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn transform(&self, x: &Tensor<T>) -> crate::error::MlResult<Self::Output> {
        let y = x
            .broadcast_sub(&self.min)?
            .broadcast_div(&self.delta)?;
        Ok(y)
    }
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
