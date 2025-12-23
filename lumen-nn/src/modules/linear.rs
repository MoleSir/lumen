use lumen_core::{FloatDType, NumDType, Tensor};
use crate::init::Initialize;

use super::{Module, ModuleVisitor};

pub fn linear<T: FloatDType>(in_dim: usize, out_dim: usize, bias: bool, init: &Initialize<T>) -> lumen_core::Result<Linear<T>> {
    let weight = init.init((out_dim, in_dim))?;
    let bias = if bias {
        let zero_init = Initialize::zeros();
        Some(zero_init.init((out_dim,))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

pub struct Linear<T: NumDType> {
    pub weight: Tensor<T>,  // (out_features, in_features)
    pub bias: Option<Tensor<T>>, // (out_features)
}

impl<T: NumDType> Linear<T> {
    pub fn new(weight: Tensor<T>, bias: Option<Tensor<T>>) -> Self {
        Self { weight, bias }
    }
    
    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // input: (batch, in_features)
        let output = input.matmul(&self.weight.transpose_last()?)?;
        match self.bias.as_ref() {
            Some(bias) => output.broadcast_add(bias),
            None => Ok(output)
        }
    }
}

impl<T: NumDType> Module<T> for Linear<T> {
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) {
        visitor.enter_module("weight");
        visitor.visit(&self.weight);
        visitor.exit_module("weight");

        if let Some(bias) = self.bias.as_ref() {
            visitor.enter_module("bias");
            visitor.visit(bias);
            visitor.exit_module("bias");
        }
    }
}

#[cfg(test)]
mod test {
    use crate::init::Initialize;
    use crate::modules::linear;
    use super::Module;

    #[test]
    fn test_module() {
        let l = linear::<f32>(100, 20, false, &Initialize::standard_normal()).unwrap();
        println!("{}", l.param_count());
        assert_eq!(l.param_count(), 2000);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }

        let l = linear::<f64>(100, 20, true, &Initialize::zeros()).unwrap();
        println!("{}", l.param_count());
        assert_eq!(l.param_count(), 2020);
        let params = l.named_params();
        for (name, _) in params {
            println!("{}", name);
        }
    }
}