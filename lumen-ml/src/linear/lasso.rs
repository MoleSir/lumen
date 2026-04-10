use lumen_core::{FloatDType, NumDType, Tensor};
use crate::{error::MlResult, pipeline::{PredictFit, PredictModel}, utils};

pub struct LassoRegression<T: FloatDType> {
    pub n_iter: usize, 
    pub learning_rate: T,
    pub alpha: T,
}

pub struct LassoRegressionModel<T: NumDType> {
    pub weights: Tensor<T>,
    pub bias: T,
    pub alpha: T,
}

impl<T: FloatDType> Default for LassoRegression<T> {
    fn default() -> Self {
        LassoRegression { n_iter: 1000, learning_rate: T::from_f64(0.01), alpha: T::from_f64(0.1) }
    }
}

impl<T: FloatDType> PredictFit for LassoRegression<T> {
    type Input  = Tensor<T>;
    type Output = Tensor<T>;
    type Model = LassoRegressionModel<T>;

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
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y)?;

        let y = y.unsqueeze(1)?;
        let weights = Tensor::<T>::zeros((n_features, 1))?; 
        let mut bias = T::ZERO;

        // train model
        /*
            y_pred = w0 * x0 + w1 * x1 + ... + b
            loss = (y_pred - y)^2 + \alpha \sum |w|

            dloss/dy_pred = 2 * (y_pred - y) + \alpha \sum |w|

            dloss/dwi =  2 * (y_pred - y) * xi + \alpha * w.sign()
            dloss/db = 2 * (y_pred - y)
        */
        let two = T::ONE + T::ONE;
        let x_t = x.transpose_last()?; // (n_features, n_samples)

        for _ in 0..self.n_iter {
            let y_pred = x.matmul(&weights)? + bias; // (n_samples, 1)
            let y_pred_grad = two * (y_pred - &y); // (n_samples, 1)
            // (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&y_pred_grad)? / T::from_usize(n_samples);
            let b_grad = y_pred_grad.mean_all()?.to_scalar()?;

            let l1_grad = self.alpha * weights.sign()?;
            w_grad.add_(l1_grad)?;
            
            w_grad.mul_(self.learning_rate)?;
            let b_grad = self.learning_rate * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LassoRegressionModel { weights, bias, alpha: self.alpha })
    }
}

impl<T: FloatDType> PredictModel for LassoRegressionModel<T> {
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

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;

    use crate::{linear::LassoRegression, pipeline::PredictFit};

    #[test]
    fn test_gd_1d() {
        const N_SAMPLES: usize = 50;
        const W: f64 = 3.0;
        const B: f64 = 2.5;
        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES,)).unwrap();
        let y_train = W * &x_train + B;
        let x_train = x_train.unsqueeze(1).unwrap();
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,)).unwrap();

        let trainer = LassoRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", model.weights);
        println!("{}", model.bias);
    }

    #[test]
    fn test_gd_nd() {
        const N_SAMPLES: usize = 50;
        const N_FEATURES: usize = 5;
        let weight = Tensor::rand(-2.0, 3.0, (N_FEATURES, 1)).unwrap();
        const B: f64 = 2.5;

        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES, N_FEATURES)).unwrap();
        let y_train = x_train.matmul(&weight).unwrap().squeeze(1).unwrap() + B;
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,)).unwrap();

        println!("{}", x_train.shape());
        println!("{}", y_train.shape());

        let trainer = LassoRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", weight);
        println!("{}", model.weights);
        println!("{}", model.bias);
    }
}