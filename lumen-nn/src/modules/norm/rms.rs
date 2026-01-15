use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{init::Init, ModuleInit, NnCtxError, NnResult, Parameter};
use crate::functional as F;

#[derive(Module)]
pub struct RMSNorm<T: FloatDType> {
    pub weight: Parameter<T>,
    #[module(skip)]
    pub variance_epsilon: T,
}

#[derive(derive_new::new)]
pub struct RMSNormConfig {
    normalized_shape: usize,
    norm_eps: f64, 
}

impl<T: FloatDType> ModuleInit<T> for RMSNorm<T> {
    type Config = RMSNormConfig;
    type Error = NnCtxError;

    fn init(config: &RMSNormConfig, init: Option<Init<T>>) -> NnResult<Self> {
        let init = init.unwrap_or(Init::ones());
        let weight = init.init((config.normalized_shape,))?;
        let variance_epsilon = T::from_f64(config.norm_eps);
        Ok(Self { 
            weight: Parameter::new(weight), 
            variance_epsilon 
        })
    }
}

impl<T: FloatDType> RMSNorm<T> {
    pub fn new(normalized_shape: usize, norm_eps: f64, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&RMSNormConfig::new(normalized_shape, norm_eps), init)
    }

    #[inline]
    pub fn forward(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        F::rms_norm(input, &self.weight, self.variance_epsilon)
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use super::RMSNorm;

    #[test]
    fn test1() {
        let ln = RMSNorm::<f64>::new(10, 1e-5, None).unwrap();
        let input = Tensor::<f64>::randn(0.0, 1.0, (2, 5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap());
    }
}