use lumen_core::{FloatDType, Tensor};
use crate::{MlResult, PredictFit, PredictModel};

pub struct GradientBoostRegressor<T, F> {
    pub fitter: F,
    pub n_estimators: usize,
    pub learning_rate: T,
    pub loss: Loss,
}

#[derive(Clone, Copy, Debug)]
pub enum Loss {
    Mse,
    Mae,
}

pub struct GradientBoostRegressorModel<T, F> {
    pub models: Vec<F>,
    pub learning_rate: T,
}

impl<T: FloatDType, F> PredictFit for GradientBoostRegressor<T, F> 
where 
    F: PredictFit<Input = Tensor<T>, Output = Tensor<T>>
{
    type Model = GradientBoostRegressorModel<T, F::Model>;
    type Input = Tensor<T>;
    type Output = Tensor<T>;

    fn fit(&self, x: &Tensor<T>, y: &Tensor<T>) -> crate::MlResult<Self::Model> {
        let y_pred = y.zeros_like()?;
        let mut models = vec![];

        for _ in 0..self.n_estimators {
            // MSE => Loss = (pred - y)
            // 负梯度 => y - pred
            let grad = self.loss.gradient(y, &y_pred)?;
            grad.mul_(T::from_f64(-1.0))?;

            let model = self.fitter.fit(x, &grad)?;

            let update = model.predict(x)?;
            update.mul_(self.learning_rate)?;
            
            y_pred.add_(&update)?;

            models.push(model);
        }

        Ok(GradientBoostRegressorModel { models, learning_rate: self.learning_rate })
    }
}

impl<T: FloatDType, M> PredictModel for GradientBoostRegressorModel<T, M> 
where 
    M: PredictModel<Input = Tensor<T>, Output = Tensor<T>>
{
    type Input = M::Input;
    type Output = M::Output;

    fn predict(&self, x: &Self::Input) -> crate::MlResult<Self::Output> {
        let y = self.models[0].predict(x)?;   
        y.mul_(self.learning_rate)?;
        for model in self.models.iter().skip(1) {
            let model_y = model.predict(x)?;
            model_y.mul_(self.learning_rate)?;
            y.add_( model_y )?;
        }
        Ok(y)
    }
}

impl Loss {
    pub fn gradient<T: FloatDType>(&self, y: &Tensor<T>, y_pred: &Tensor<T>) -> MlResult<Tensor<T>> {
        match self {
            Loss::Mse => {
                // Loss = (y_pred - y)^2
                // dLoss = 2 * (y_pred - y)
                let grad = y_pred - y;  // y_pred - y
                grad.mul_(T::from_f64(2.0))?;
                Ok(grad)
            }
            Loss::Mae => {
                // Loss = |y_pred - y|
                let grad = y_pred - y;
                grad.sign_()?;
                Ok(grad)
            }
        }
    }
}