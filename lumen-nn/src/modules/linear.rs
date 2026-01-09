use lumen_core::{FloatDType, NumDType, Tensor};
use lumen_macros::Module;
use crate::init::Initialize;

pub fn linear<T: FloatDType>(in_dim: usize, out_dim: usize, bias: bool, init: &Initialize<T>) -> lumen_core::Result<Linear<T>> {
    let weight = init.init((out_dim, in_dim))?;
    let bias = if bias {
        let zero_init = Initialize::zeros();
        Some(zero_init.init((out_dim,))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, in_dim, out_dim))
}

#[derive(Module)]
pub struct Linear<T: NumDType> {
    pub weight: Tensor<T>,  // (out_features, in_features)
    pub bias: Option<Tensor<T>>, // (out_features)

    #[module(skip)]
    pub in_dim: usize,
    #[module(skip)]
    pub out_dim: usize,
}

impl<T: NumDType> Linear<T> {
    pub fn new(weight: Tensor<T>, bias: Option<Tensor<T>>, in_dim: usize, out_dim: usize) -> Self {
        Self { weight, bias, in_dim, out_dim }
    }
    
    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        crate::functional::linear(input, &self.weight, self.bias.as_ref())
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    use crate::init::Initialize;
    use crate::modules::{linear, Module};

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

    #[test]
    fn test_forward() {
        let l = linear::<f32>(30, 20, false, &Initialize::standard_normal()).unwrap();

        let input = Tensor::<f32>::rand(0.0, 1.0, (1, 30, 30)).unwrap();
        println!("{}", l.forward(&input).unwrap().shape());
    }
}