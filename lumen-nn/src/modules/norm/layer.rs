use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Init, ModuleInit, NnCtxError, NnResult};

#[derive(Module)]
pub struct LayerNorm<T: FloatDType> {
    pub weight: Tensor<T>,
    pub bias: Tensor<T>,

    #[module(skip)]
    pub normalized_shape: usize,
    #[module(skip)]
    pub variance_epsilon: T,
}

#[derive(derive_new::new)]
pub struct LayerNormConfig {
    normalized_shape: usize,
    norm_eps: f64, 
}

impl<T: FloatDType> ModuleInit<T> for LayerNorm<T> {
    type Config = LayerNormConfig;
    type Error = NnCtxError;

    fn init(config: &LayerNormConfig, init: Option<Init<T>>) -> NnResult<Self> {
        let weight = init.unwrap_or(Init::ones()).init((config.normalized_shape,))?;
        let bias = init.unwrap_or(Init::zeros()).init((config.normalized_shape,))?;
        let variance_epsilon = T::from_f64(config.norm_eps);
        Ok(Self { weight, bias, normalized_shape: config.normalized_shape, variance_epsilon })
    }
}

impl<T: FloatDType> LayerNorm<T> {
    pub fn new(normalized_shape: usize, norm_eps: f64, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&LayerNormConfig::new(normalized_shape, norm_eps), init)
    }

    pub fn forward(&self, x: &Tensor<T>) -> NnResult<Tensor<T>> {
        // (xxx, normalized_shape) => (xxx, 1)
        let mean = x.mean_keepdim(D::Minus1)?;
        let var = x.var_keepdim(D::Minus1)?;

        let x_normalized = x.broadcast_sub(&mean)?.broadcast_div(&(var + self.variance_epsilon))?;

        let res = x_normalized
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;

        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use super::LayerNorm;

    #[test]
    fn test1() {
        let ln = LayerNorm::<f64>::new(10, 1e-5, None).unwrap();
        let input = Tensor::<f64>::randn(0.0, 1.0, (2, 5, 10)).unwrap();    
        println!("{}", ln.forward(&input).unwrap());
    }
}