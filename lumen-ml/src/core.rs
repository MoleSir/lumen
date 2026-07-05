use crate::MlResult;

pub trait PredictFit {
    type Input;
    type Output;
    type Model: PredictModel<Input = Self::Input, Output = Self::Output>;
    
    fn fit(&self, x: &Self::Input, y: &Self::Output) -> MlResult<Self::Model>;
    
    fn fit_predict(&self, x: &Self::Input, y: &Self::Output) -> MlResult<Self::Output> {
        let model = self.fit(x, y)?;
        let y_pred = model.predict(x)?;
        Ok(y_pred)
    }
}

pub trait PredictFitWithWeight : PredictFit {
    type Weight;
    fn fit_with_weight(&self, x: &Self::Input, y: &Self::Output, weight: &Self::Weight) -> MlResult<Self::Model>;
}

pub trait PredictModel {
    type Input;
    type Output;
    fn predict(&self, x: &Self::Input) -> MlResult<Self::Output>;
}

pub trait TransformFit {
    type Input;
    type Output;
    type Model: TransformModel<Input = Self::Input, Output = Self::Output>;

    fn fit(&self, x: &Self::Input) -> MlResult<Self::Model>;

    fn fit_transform(&self, x: &Self::Input) -> MlResult<Self::Output> {
        let model = self.fit(x)?;
        let x_trans = model.transform(x)?;
        Ok(x_trans)
    }
}

pub trait TransformModel {
    type Input;
    type Output;
    fn transform(&self, x: &Self::Input) -> MlResult<Self::Output>;
}
