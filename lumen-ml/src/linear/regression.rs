use lumen_core::{FloatDType, NumDType, Tensor};
use crate::{core::{PredictFit, PredictModel}, error::MlResult, utils, PredictFitWithWeight};

pub enum LinearRegression<T: FloatDType> {
    GradientDescent { n_iter: usize, learning_rate: T, },
    NormalEquations,
}

pub struct LinearRegressionModel<T: NumDType> {
    pub weights: Tensor<T>,
    pub bias: T,
}

impl<T: FloatDType> Default for LinearRegression<T> {
    fn default() -> Self {
        LinearRegression::GradientDescent { n_iter: 1000, learning_rate: T::from_f64(0.01) }
    }
}

impl<T: FloatDType> PredictFit for LinearRegression<T> {
    type Input  = Tensor<T>;
    type Output = Tensor<T>;
    type Model = LinearRegressionModel<T>;

    /// fit a linear regression model
    /// $$
    /// y = w x + b
    /// $$
    /// 
    /// ## Args
    /// - `x`: (n_samples, n_features) 
    /// - `y`: (n_samples,)
    /// 
    /// ## Return
    /// - linear gression model
    fn fit(&self, x: &Tensor<T>, y: &Tensor<T>) -> MlResult<Self::Model> {
        match self {
            Self::GradientDescent { n_iter, learning_rate } => Self::fit_gd(x, y, None, *n_iter, *learning_rate),
            Self::NormalEquations => Self::fit_ne(x, y),
        }
    }
}

impl<T: FloatDType> PredictFitWithWeight for LinearRegression<T> {
    type Weight = Tensor<T>;

    fn fit_with_weight(&self, x: &Self::Input, y: &Self::Output, weight: &Self::Weight) -> MlResult<Self::Model> {
        match self {
            Self::GradientDescent { n_iter, learning_rate } => Self::fit_gd(x, y, Some(weight), *n_iter, *learning_rate),
            Self::NormalEquations => Self::fit_ne(x, y),
        }
    }
}

impl<T: FloatDType> PredictModel for LinearRegressionModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    /// ## Args:
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Return
    /// - `y`: (n_samples, )
    fn predict(&self, x: &Tensor<T>) -> MlResult<Tensor<T>> {
        let y = x.matmul(&self.weights)?;
        y.add_(self.bias)?;
        Ok(y.squeeze(1)?)
    }
}

impl<T: FloatDType> LinearRegression<T> {
    #[allow(unused)]
    fn fit_ne(x: &Tensor<T>, y: &Tensor<T>) -> MlResult<LinearRegressionModel<T>> {
        unimplemented!("fit with Normal Equations")
    }

    /// - `x`: (n_samples, n_features)
    /// - `y`: (n_samples),
    /// - `sample_weight`: Option<(n_samples)>,    
    fn fit_gd(x: &Tensor<T>, y: &Tensor<T>, sample_weight: Option<&Tensor<T>>, n_iter: usize, learning_rate: T) -> MlResult<LinearRegressionModel<T>> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, sample_weight)?;
        
        // 初始化参数
        let weights = Tensor::zeros((n_features, 1))?;
        let mut bias = T::ZERO;

        let y = y.unsqueeze(1)?; // (n_samples, 1)
        let x_t = x.transpose_last()?;
        let two = T::two();

        // (n_samples, 1)
        let sample_weight = match sample_weight {
            Some(w) => w.unsqueeze(1)?,
            None => Tensor::ones((n_samples, 1))?,
        };
        let weight_sum = sample_weight.sum_all()?.to_scalar()?;

        for _ in 0..n_iter {
            /*
        
                y_pred = x @ w + b

                loss = sw * (y_pred - y)^2

                dloss = 2 * sw * (y_pred - y)

                dwi = 2 * sw * (y_pred - y) * xi

                db = 2 * sw * (y_pred - y)

                其中 dwi 是某个 wi 的梯度，而多个 wi 就是向量的数乘

                dw = 2 * sw * (y_pred - y) * \vec x 

                上述推导是对每个 sample，如何要推广到所有 samples，就是求和后平均，
                例如计算的 y_pred 已经是一个向量形式，那么对 b 比较容易，直接对 `2 * (y_pred - y)` 计算求和即可（因为这个算出来还是向量）
                而对 w 稍微复杂一些，单个 sample 是向量数乘，多个 sample 可以拓展到矩阵向量乘，

                2 * x.T @ sw * (y_pred - y)

                x.T 每次取出一行（所有 sample 的某个特征）和所有的 (y_pred - y) 分别相乘法后相加。相当于每次把所有批次的一个特征计算好
                最后恰好得到所有特征
                
            */

            // (n_samples, n_features) @ (n_features, 1) => (n_samples, 1)
            let y_pred = x.matmul(&weights)?.add_(bias)?;

            // (n_samples, 1)
            let dloss = (&y_pred - &y).mul_(&sample_weight)?.mul_(two)?;

            // (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&dloss)?.div_(weight_sum)?;
            let b_grad = dloss.sum_all()?.div_(weight_sum)?.to_scalar()?;

            
            w_grad.mul_(learning_rate)?;
            let b_grad = learning_rate * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LinearRegressionModel { weights, bias })
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;

    use crate::{datasets::{make_regression, RegressionOption}, linear::LinearRegression, core::PredictFit};

    #[test]
    fn test_gd_1d() {
        const N_SAMPLES: usize = 50;
        const W: f64 = 3.0;
        const B: f64 = 2.5;
        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES,)).unwrap();
        let y_train = W * &x_train + B;
        let x_train = x_train.unsqueeze(1).unwrap();
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,)).unwrap();

        let trainer = LinearRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", model.weights);
        println!("{}", model.bias);
    }

    #[test]
    fn test_gd_nd() {
        const N_SAMPLES: usize = 50;
        const N_FEATURES: usize = 5;
        let train_data = make_regression::<f64>(N_SAMPLES, N_FEATURES, RegressionOption::default()).unwrap();
        let x_train = train_data.x;
        let y_train = train_data.y;
        let weight_bias = train_data.coef;

        let trainer = LinearRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", weight_bias);
        println!("{}", model.weights);
        println!("{}", model.bias);
    }
}