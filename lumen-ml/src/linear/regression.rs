use lumen_core::{NumDType, Tensor};

pub enum LinearRegressionTrainer {
    GradientDescent { n_iter: usize, learning_rate: f64, },
    NormalEquations,
}

impl Default for LinearRegressionTrainer {
    fn default() -> Self {
        LinearRegressionTrainer::GradientDescent { n_iter: 1000, learning_rate: 0.01 }
    }
}

impl LinearRegressionTrainer {
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
    pub fn fit<T: NumDType>(&self, x: &Tensor<T>, y: &Tensor<T>) -> lumen_core::Result<LinearRegression<T>> {
        match self {
            Self::GradientDescent { n_iter, learning_rate } => Self::fit_gd(x, y, *n_iter, *learning_rate),
            Self::NormalEquations => Self::fit_ne(x, y),
        }
    }

    #[allow(unused)]
    fn fit_ne<T: NumDType>(x: &Tensor<T>, y: &Tensor<T>) -> lumen_core::Result<LinearRegression<T>> {
        unimplemented!("fit with Normal Equations")
    }

    fn fit_gd<T: NumDType>(x: &Tensor<T>, y: &Tensor<T>, n_iter: usize, learning_rate: f64) -> lumen_core::Result<LinearRegression<T>> {
        let (n_samples, n_features) = x.dims2()?;
        let n_samples_y = y.dims1()?;
        if n_samples != n_samples_y {
            lumen_core::bail!("todo");
        }

        let y = y.unsqueeze(1)?;
        let weights = Tensor::<T>::zeros((n_features, 1))?; 
        let mut bias = T::ZERO;

        // train model
        /*
            y_pred = w0 * x0 + w1 * x1 + ... + b
            loss = (y_pred - y)^2

            dloss/dy_pred = 2 * (y_pred - y)

            dloss/dwi =  2 * (y_pred - y) * xi
            dloss/db = 2 * (y_pred - y)
        */
        let two = T::ONE + T::ONE;
        let lr = T::from_f64(learning_rate);
        let x_t = x.transpose_last()?; // (n_features, n_samples)

        for _ in 0..n_iter {
            let y_pred = x.matmul(&weights)? + bias; // (n_samples, 1)
            let y_pred_grad = two * (y_pred - &y); // (n_samples, 1)
            // (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&y_pred_grad)? / T::from_usize(n_samples);
            let b_grad = y_pred_grad.mean_all()?.to_scalar()?;
            
            w_grad.mul_(lr)?;
            let b_grad = lr * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LinearRegression { weights, bias })
    }
}

pub struct LinearRegression<T: NumDType> {
    pub weights: Tensor<T>,
    pub bias: T,
}

impl<T: NumDType> LinearRegression<T> {
    /// ## Args:
    /// - `x`: (n_samples, n_features)
    /// 
    /// ## Return
    /// - `y`: (n_samples, )
    pub fn predict(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let y = x.matmul(&self.weights)?;
        y.add_(self.bias)?;
        Ok(y.squeeze(1)?)
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::Tensor;

    use crate::linear::LinearRegressionTrainer;

    #[test]
    fn test_gd_1d() {
        const N_SAMPLES: usize = 50;
        const W: f64 = 3.0;
        const B: f64 = 2.5;
        let x_train = Tensor::rand(-1.0, 1.0, (N_SAMPLES,)).unwrap();
        let y_train = W * &x_train + B;
        let x_train = x_train.unsqueeze(1).unwrap();
        let y_train = y_train + 0.1 * Tensor::randn(0.0, 1.0, (N_SAMPLES,)).unwrap();

        let trainer = LinearRegressionTrainer::default();
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

        let trainer = LinearRegressionTrainer::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();
        println!("{}", weight);
        println!("{}", model.weights);
        println!("{}", model.bias);
    }
}