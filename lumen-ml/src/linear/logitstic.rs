use lumen_core::{FloatDType, Tensor};

pub struct LogisticRegressionTrainer {
    pub n_iter: usize, 
    pub learning_rate: f64,
}

impl Default for LogisticRegressionTrainer {
    fn default() -> Self {
        LogisticRegressionTrainer { n_iter: 1000, learning_rate: 0.1 } 
    }
}

impl LogisticRegressionTrainer {
    /// fit a logistic regression model (Binary Classification)
    /// $$
    /// z = w x + b
    /// y\_pred = \frac{1}{1 + e^{-z}}
    /// $$
    /// 
    /// ## Args
    /// - `x`: (n_samples, n_features) 
    /// - `y`: (n_samples,) , boolean values for binary classes
    /// 
    /// ## Return
    /// - logistic regression model
    pub fn fit<T: FloatDType>(&self, x: &Tensor<T>, y: &Tensor<bool>) -> lumen_core::Result<LogisticRegression<T>> {
        let (n_samples, n_features) = x.dims2()?;
        let n_samples_y = y.dims1()?;
        if n_samples != n_samples_y {
            lumen_core::bail!("The number of samples in x and y must be equal");
        }

        let y_float = y.cast::<T>()?.unsqueeze(1)?; 
        let weights = Tensor::<T>::zeros((n_features, 1))?; 
        let mut bias = T::ZERO;

        let lr = T::from_f64(self.learning_rate);
        let n_samples_t = T::from_usize(n_samples);
        let x_t = x.transpose_last()?; // (n_features, n_samples)

        // train model
        /*
            z = XW + b
            y_pred = sigmoid(z) = 1 / (1 + exp(-z))
            
            Loss (BCE) = - y*log(y_pred) - (1-y)*log(1-y_pred)
            
            dloss/dz = y_pred - y
            dloss/dw = X^T @ (y_pred - y) / N
            dloss/db = mean(y_pred - y)
        */
        for _ in 0..self.n_iter {
            // forward pass
            let z = x.matmul(&weights)? + bias; // (n_samples, 1)            
            let y_pred = z.sigmoid()?;

            // backward
            let y_pred_grad = y_pred - &y_float; // (n_samples, 1)
            
            // W_grad: (n_features, n_samples) @ (n_samples, 1) => (n_features, 1)
            let w_grad = x_t.matmul(&y_pred_grad)? / n_samples_t;
            let b_grad = y_pred_grad.mean_all()?.to_scalar()?;
            
            // Update
            w_grad.mul_(lr)?;
            let b_grad = lr * b_grad;
            weights.sub_(w_grad)?;
            bias -= b_grad;
        }

        Ok(LogisticRegression { weights, bias })
    }
}

pub struct LogisticRegression<T: FloatDType> {
    pub weights: Tensor<T>,
    pub bias: T,
}

impl<T: FloatDType> LogisticRegression<T> {
    pub fn predict_proba(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let z = x.matmul(&self.weights)?;
        z.add_(self.bias)?;
        z.sigmoid_()?;
        Ok(z.squeeze(1)?)
    }

    pub fn predict(&self, x: &Tensor<T>, thre: T) -> lumen_core::Result<Tensor<bool>> {
        let probs = self.predict_proba(x)?;        
        let preds = probs.ge(thre)?; 
        
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::IndexOp;
    use crate::{datasets::load_iris, model_selection::train_test_split};
    use super::LogisticRegressionTrainer;

    #[test]
    fn test_iris() {
        let iris = load_iris::<f32>().unwrap();
        let x = iris.data;
        let y = iris.target;
        let x = x.index((.., ..2)).unwrap().contiguous().unwrap();
        let y = y.eq(0).unwrap();
        let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.3).unwrap();
        // println!("{}", x_train.shape());
        // println!("{}", x_test.shape());
        // println!("{}", y_train.shape());
        // println!("{}", y_test.shape());

        let trainer = LogisticRegressionTrainer::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test, 0.5).unwrap();

        let n_correct = y_pred.xor(y_test).unwrap().false_count().unwrap();
        println!("Correct count: {}", n_correct);
        println!("Accurry: {}%", (n_correct as f64 / x_test.dims()[0] as f64) * 100.0);
    }
}