use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{init::Init, NnCtxError, NnResult};
use super::ModuleInit;

/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
#[derive(Module)]
pub struct Linear<T: FloatDType> {
    pub weight: Tensor<T>,  // (out_features, in_features)
    pub bias: Option<Tensor<T>>, // (out_features)

    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

impl<T: FloatDType> Linear<T> { 
    #[inline]
    pub fn new(in_features: usize, out_features: usize, bias: bool, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&LinearConfig::new(in_features, out_features, bias), init)
    }
    
    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        crate::functional::linear(input, &self.weight, self.bias.as_ref())
    }
}

impl<T: FloatDType> ModuleInit<T> for Linear<T> {
    type Config = LinearConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let weight = init
            .unwrap_or_else(|| {
                let gain = T::one() / T::from_f64(3.0).sqrt();
                Init::kaiming_uniform(gain, false)
            })
            .init_with((config.out_features, config.in_features), config.in_features, config.out_features)?;

        let bias = if config.bias {
            let zero_init = Init::zeros();
            Some(zero_init.init((config.out_features,))?)
        } else {
            None
        };

        Ok(Self { weight, bias, in_features: config.in_features, out_features: config.out_features })
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    use crate::init::Init;
    use crate::modules::Module;
    use crate::Linear;

    #[test]
    fn test_module() {
        let l = Linear::<f32>::new(100, 20, false, None).unwrap();
        println!("{}", l.param_count());
        assert_eq!(l.param_count(), 2000);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }

        let l = Linear::<f32>::new(100, 20, true, None).unwrap();
        println!("{}", l.param_count());
        assert_eq!(l.param_count(), 2020);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }
    }

    #[test]
    fn test_forward() {
        let l = Linear::<f32>::new(30, 20, false, Some(Init::standard_normal())).unwrap();

        let input = Tensor::<f32>::rand(0.0, 1.0, (1, 30, 30)).unwrap();
        println!("{}", l.forward(&input).unwrap().shape());
    }
}