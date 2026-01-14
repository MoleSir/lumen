use std::ops::{Deref, DerefMut};

use lumen_core::{FloatDType, Tensor};
use crate::{init::Init, Module, ModuleInit, ModuleVisitor, ModuleVisitorMut, NnCtxError, NnResult, TensorVisitor, TensorVisitorMut};

#[derive(Clone)]
pub struct Parameter<T: FloatDType>(Tensor<T>);

impl<T: FloatDType> Deref for Parameter<T> {
    type Target = Tensor<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
} 

impl<T: FloatDType> DerefMut for Parameter<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
} 

impl<T: FloatDType> Parameter<T> {
    pub fn new(tensor: Tensor<T>) -> Self {
        tensor.set_requires_grad(true);
        Self(tensor)
    }

    pub fn tensor(&self) -> &Tensor<T> {
        &self.0
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.0
    }
}

impl<T: FloatDType> Module<T> for Parameter<T> {
    #[inline]
    fn visit_param<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param(self.tensor())
    }

    #[inline]
    fn visit_param_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param_mut(self.tensor_mut())
    }

    #[allow(unused_variables)]
    fn visit_buffer<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_buffer_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[allow(unused_variables)]
    fn visit_state<Visitor: TensorVisitor<T>>(&self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param(self.tensor())
    }

    #[allow(unused_variables)]
    fn visit_state_mut<Visitor: TensorVisitorMut<T>>(&mut self, visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        visitor.visit_param_mut(self.tensor_mut())
    }

    #[inline]
    fn visit_module<Visitor: ModuleVisitor<T>>(&self, _visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[inline]
    fn visit_module_mut<Visitor: ModuleVisitorMut<T>>(&mut self, _visitor: &mut Visitor) -> Result<(), Visitor::Error> {
        Ok(())
    }

    #[inline]
    fn set_train(&mut self, mode: bool) {
        self.0.set_requires_grad(mode);
    }
}


/// Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
#[derive(Module, Clone)]
#[module(display = "display")]
pub struct _Linear<T: FloatDType> {
    pub weight: Parameter<T>,  // (out_features, in_features)
    pub bias: Option<Parameter<T>>, // (out_features)

    #[module(skip)]
    pub in_features: usize,
    #[module(skip)]
    pub out_features: usize,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct _LinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

impl<T: FloatDType> _Linear<T> { 
    #[inline]
    pub fn new(in_features: usize, out_features: usize, bias: bool, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&_LinearConfig::new(in_features, out_features, bias), init)
    }
    
    pub fn forward(&self, input: &Tensor<T>) -> NnResult<Tensor<T>> {
        crate::functional::linear(input, &self.weight.tensor(), self.bias.as_ref().map(|p| p.tensor()))
    }

    fn display(&self) -> String {
        format!("in_features={}, out_features={}", self.in_features, self.out_features)
    }
}

impl<T: FloatDType> ModuleInit<T> for _Linear<T> {
    type Config = _LinearConfig;
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

        Ok(Self { weight: Parameter::new(weight), bias: bias.map(Parameter::new), in_features: config.in_features, out_features: config.out_features })
    }
}

#[cfg(test)]
mod test {
    use lumen_core::{FloatDType, Tensor};
    use lumen_macros::Module;

    use crate::init::Init;
    use crate::modules::Module;
    use crate::NnResult;

    use super::_Linear;

    #[derive(Module)]
    pub struct Net<T: FloatDType> {
        pub fc1: _Linear<T>,
        pub fc2: _Linear<T>,
        pub fc3: _Linear<T>,
    }

    impl<T: FloatDType> Net<T> {
        pub fn new() -> NnResult<Self> {
            let fc1 = _Linear::new(784, 512, true, None)?;
            let fc2 = _Linear::new(512, 256, true, None)?;
            let fc3 = _Linear::new(256, 10, true, None)?;
    
            Ok(Self { fc1, fc2, fc3 })
        }
    }

    #[test]
    fn test_mlp() {
        let net = Net::<f32>::new().unwrap();
        println!("{}", net);
        println!("{}", net.param_element_count());
        println!("{}", net.submodule_count());
        println!("{:?}", net.submodule_names());
    }

    #[test]
    fn test_module() {
        let l = _Linear::<f32>::new(100, 20, false, None).unwrap();
        assert_eq!(l.param_element_count(), 2000);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }

        let l = _Linear::<f32>::new(100, 20, true, None).unwrap();
        let ll = l.copy();

        assert_eq!(ll.param_element_count(), 2020);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }

        println!("{}", ll);

        // ll.apply_param(|p| print!("{}", p));
    }

    #[test]
    fn test_forward() {
        let l = _Linear::<f32>::new(30, 20, false, Some(Init::standard_normal())).unwrap();

        let input = Tensor::<f32>::rand(0.0, 1.0, (1, 30, 30)).unwrap();
        println!("{}", l.forward(&input).unwrap().shape());
    }
}