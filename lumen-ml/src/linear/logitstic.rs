use lumen_core::{FloatDType, Tensor};
use crate::{error::MlResult, core::{PredictFit, PredictModel}, utils};

pub struct LogisticRegression<T> {
    pub n_iter: usize, 
    pub learning_rate: T,
    pub threshold: T,
}

pub struct LogisticRegressionModel<T: FloatDType> {
    pub weights: Tensor<T>,
    pub bias: T,
    pub threshold: T,
}

impl<T: FloatDType> Default for LogisticRegression<T> {
    fn default() -> Self {
        LogisticRegression { n_iter: 1000, learning_rate: T::from_f64(0.1), threshold: T::half(), } 
    }
}

impl<T: FloatDType> PredictFit for LogisticRegression<T> {
    type Input = Tensor<T>;
    type Output = Tensor<bool>;
    type Model = LogisticRegressionModel<T>;

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
    fn fit(&self, x: &Tensor<T>, y: &Tensor<bool>) -> MlResult<Self::Model> {
        let (n_samples, n_features) = utils::validate_xy_shapes(x, y, None)?;

        let y_float = y.cast::<T>()?.unsqueeze(1)?; 
        let weights = Tensor::<T>::zeros((n_features, 1))?; 
        let mut bias = T::ZERO;

        let lr = self.learning_rate;
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

        Ok(LogisticRegressionModel { weights, bias, threshold: self.threshold })
    }
}

impl<T: FloatDType> PredictModel for LogisticRegressionModel<T> {
    type Input = Tensor<T>;
    type Output = Tensor<bool>;

    /// ## Args
    /// - `x`: (n_samples, n_features) 
    /// 
    /// ## Return
    /// - y: (n_samples,)
    fn predict(&self, x: &Tensor<T>) -> MlResult<Tensor<bool>> {        
        self.predict_threshold(x, self.threshold)
    }
}

impl<T: FloatDType> LogisticRegressionModel<T> {
    pub fn predict_proba(&self, x: &Tensor<T>) -> MlResult<Tensor<T>> {
        let z = x.matmul(&self.weights)?;
        z.add_(self.bias)?;
        z.sigmoid_()?;
        Ok(z.squeeze(1)?)
    }

    pub fn predict_threshold(&self, x: &Tensor<T>, threshold: T) -> MlResult<Tensor<bool>> {
        let probs = self.predict_proba(x)?;        
        let preds = probs.ge(threshold)?; 
        
        Ok(preds)
    }
}

#[cfg(test)]
mod tests {
    use lumen_core::IndexOp;
    use crate::{datasets::{load_iris, train_test_split}, core::{PredictFit, PredictModel}};
    use super::LogisticRegression;

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

        let trainer = LogisticRegression::default();
        let model = trainer.fit(&x_train, &y_train).unwrap();

        let y_pred = model.predict(&x_test).unwrap();

        let n_correct = y_pred.xor(y_test).unwrap().false_count().unwrap();
        println!("Correct count: {}", n_correct);
        println!("Accurry: {}%", (n_correct as f64 / x_test.dims()[0] as f64) * 100.0);
    }
}