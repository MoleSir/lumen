
use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;
use crate::{ModuleForward, NnCtxError, NnResult};

#[derive(Module)]
#[module(display = "display")]
pub struct Flatten {
    #[module(skip)]
    start_dim: D,
    #[module(skip)]
    end_dim: D,
}

impl Flatten {
    pub fn new(start_dim: D, end_dim: D) -> Self {
        Self { start_dim, end_dim }
    }

    pub fn batch() -> Self {
        Self::new(D::Index(1), D::Minus1)
    }

    pub fn forward<T: FloatDType>(&self, xs: &Tensor<T>) -> NnResult<Tensor<T>> {
        let output = xs.flatten(self.start_dim, self.end_dim)?;
        Ok(output)
    }

    fn display(&self) -> String {
        format!("start_dim={}, end_dim={}", self.start_dim, self.end_dim)
    }
}

impl<T: FloatDType> ModuleForward<T> for Flatten {
    type Error = NnCtxError;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn forward(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        Flatten::forward(self, &input)
    }
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use super::Flatten;

    #[test]
    fn test_eval_and_train() {
        let flatten = Flatten::batch();
        let x = Tensor::<f64>::rand(0.0, 1.0, (10, 3, 3)).unwrap();
        let y = flatten.forward(&x).unwrap();
        assert_eq!(y.dims(), [10, 9])
    }
}