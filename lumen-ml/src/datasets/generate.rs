use derive_builder::Builder;
use lumen_core::{FloatDType, IndexOp, Tensor};
use crate::error::MlResult;

/// Make a regression dataset
/// 
/// ## Args
/// - `n_samples`
/// - `n_features`
/// - `option`: RegressionOption
/// 
/// ## Returns
/// - `x_train`: (n_samples, n_features)
/// - `y_train`: (n_samples,)
/// - `coef`: (n_features + 1,)
pub fn make_regression<T: FloatDType>(n_samples: usize, n_features: usize, option: RegressionOption) -> MlResult<RegressionData<T>> {
    option.generate(n_samples, n_features)
}

#[derive(Builder)]
#[builder(pattern = "owned")] // 允许 builder().noise(0.1).generate(...) 链式调用
pub struct RegressionOption {
    #[builder(default)]
    pub mean: f64,
    #[builder(default = "1.0")]
    pub std: f64,
    #[builder(default = "0.1")]
    pub noise: f64,
    #[builder(default = "None")]
    pub seed: Option<u32>,
}

pub struct RegressionData<T: FloatDType> {
    pub x: Tensor<T>,
    pub y: Tensor<T>,
    pub coef: Tensor<T>,
}

impl RegressionOption {
    pub fn generate<T: FloatDType>(self, n_samples: usize, n_features: usize) -> MlResult<RegressionData<T>> {
        let mean = T::from_f64(self.mean);
        let std_dev = T::from_f64(self.std);
        let noise = T::from_f64(self.noise);

        let weight_bias = Tensor::randn(mean, std_dev, (n_features + 1,))?;
        let weight = weight_bias.index(..n_features)?;
        let bias = weight_bias.index(n_features)?.to_scalar()?;

        let x = Tensor::randn(mean, std_dev, (n_samples, n_features))?;
        let y = x.matmul(&weight.unsqueeze(1)?)?.squeeze(1)?;
        
        y.add_(bias)?;
        if self.noise > 0.0 {
            y.add_(noise * Tensor::randn(T::ZERO, T::ONE, (n_samples,))?)?;
        }

        Ok(RegressionData { x, y, coef: weight_bias })
    }
}

impl Default for RegressionOption {
    fn default() -> Self {
        RegressionOptionBuilder::default().build().expect("build error")
    }
}
