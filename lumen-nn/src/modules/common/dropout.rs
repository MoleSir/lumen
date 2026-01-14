use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use crate::functional as F;

#[derive(Module)]
#[module(display = "display")]
#[module(train = "set_train")]
pub struct Dropout<T: FloatDType> {
    #[module(skip)]
    drop_p: T,
    #[module(skip)]
    train: bool
}

impl<T: FloatDType> Dropout<T> {
    pub fn new(drop_p: T) -> Self {
        Self { drop_p, train: true }
    }

    pub fn train(drop_p: T) -> Self {
        Self { drop_p, train: true }
    }

    pub fn eval(drop_p: T) -> Self {
        Self { drop_p, train: false }
    }

    pub fn forward(&self, xs: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        if self.train {
            F::dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }

    fn display(&self) -> String {
        format!("drop_p={}", self.drop_p)
    }

    fn set_train(&mut self, mode: bool) {
        self.train = mode;
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;

    use crate::Module;

    use super::Dropout;

    #[test]
    fn test_eval_and_train() {
        let mut dropout = Dropout::train(0.2);
        let xs = Tensor::<f64>::rand(0.0, 1.0, (3, 3)).unwrap();
        println!("{}", xs);
        println!("{}", dropout.forward(&xs).unwrap());

        dropout.eval();
        println!("{}", dropout.forward(&xs).unwrap());
        assert!(dropout.forward(&xs).unwrap().allclose(&xs, 1e-5, 8e-8));
    }
}