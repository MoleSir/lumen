use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::{init::Init, NnCtxError, NnResult};
use super::ModuleInit;

/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
#[derive(Module, Clone)]
#[module(display = "display")]
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

    fn display(&self) -> String {
        format!("in_features={}, out_features={}", self.in_features, self.out_features)
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
    use lumen_core::{FloatDType, Tensor};
    use lumen_macros::Module;

    use crate::init::Init;
    use crate::modules::Module;
    use crate::{Linear, NnResult};

    #[derive(Module)]
    pub struct Net<T: FloatDType> {
        pub fc1: Linear<T>,
        pub fc2: Linear<T>,
        pub fc3: Linear<T>,
    }

    impl<T: FloatDType> Net<T> {
        pub fn new() -> NnResult<Self> {
            let fc1 = Linear::new(784, 512, true, None)?;
            let fc2 = Linear::new(512, 256, true, None)?;
            let fc3 = Linear::new(256, 10, true, None)?;
    
            Ok(Self { fc1, fc2, fc3 })
        }
    }

    #[test]
    fn test_mlp() {
        let net = Net::<f32>::new().unwrap();
        println!("{}", net);
        println!("{}", net.param_count());
        println!("{}", net.submodule_count());
        println!("{:?}", net.submodule_names());
    }

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
        let ll = l.copy();

        println!("{}", ll.param_count());
        assert_eq!(ll.param_count(), 2020);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }

        // println!("{}", ll);
    }

    #[test]
    fn test_forward() {
        let l = Linear::<f32>::new(30, 20, false, Some(Init::standard_normal())).unwrap();

        let input = Tensor::<f32>::rand(0.0, 1.0, (1, 30, 30)).unwrap();
        println!("{}", l.forward(&input).unwrap().shape());
    }
}