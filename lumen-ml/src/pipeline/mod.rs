#[cfg(test)]
mod tests;
use crate::{MlResult, PredictFit, PredictModel, TransformFit, TransformModel};

// ============================================================================= //
//                 Pipline
// ============================================================================= //

pub struct Pipeline<H, T> {
    head: H,
    tail: T,
}

pub struct PipelineModel<MH, MT> {
    head: MH,
    tail: MT,
}

impl<H, T> Pipeline<H, T> {
    pub fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }
}

impl<H, T> PipelineModel<H, T> {
    pub fn new(head: H, tail: T) -> Self {
        Self { head, tail }
    }
}

impl<H, T> Pipeline<H, T> {
    pub fn pipe<U>(self, next: U) -> Pipeline<Self, U> {
        Pipeline {
            head: self,
            tail: next
        }
    } 
}

impl<H, T> PredictFit for Pipeline<H, T> 
where 
    H: TransformFit,
    T: PredictFit<Input = H::Output>,
{
    type Input = H::Input;
    type Output = T::Output;
    type Model = PipelineModel<H::Model, T::Model>;

    fn fit(&self, x: &Self::Input, y: &Self::Output) -> MlResult<Self::Model> {
        let mh = self.head.fit(x)?;
        let x_transformed = mh.transform(x)?;
        let mt = self.tail.fit(&x_transformed, y)?;
        Ok(PipelineModel { head: mh, tail: mt })
    }
}

impl<H, T> TransformFit for Pipeline<H, T>
where
    H: TransformFit,
    T: TransformFit<Input = H::Output>,
{
    type Input = H::Input;
    type Output = T::Output;
    type Model = PipelineModel<H::Model, T::Model>;

    fn fit(&self, x: &Self::Input) -> MlResult<Self::Model> {
        let mh = self.head.fit(x)?;
        let x_transformed = mh.transform(x)?;
        let mt = self.tail.fit(&x_transformed)?;
        Ok(PipelineModel { head: mh, tail: mt })
    }
}

impl<MH, MT> PredictModel for PipelineModel<MH, MT>
where
    MH: TransformModel,
    MT: PredictModel<Input = MH::Output>,
{
    type Input = MH::Input;
    type Output = MT::Output;

    fn predict(&self, x: &Self::Input) -> MlResult<Self::Output> {
        let x_next = self.head.transform(x)?;
        self.tail.predict(&x_next)
    }
}

impl<MH, MT> TransformModel for PipelineModel<MH, MT>
where
    MH: TransformModel,
    MT: TransformModel<Input = MH::Output>,
{
    type Input = MH::Input;
    type Output = MT::Output;

    fn transform(&self, x: &Self::Input) -> MlResult<Self::Output> {
        let x_next = self.head.transform(x)?;
        self.tail.transform(&x_next)
    }
}

#[macro_export]
macro_rules! pipelines {
    ($head:expr, $tail:expr $(, $rest:expr)*) => {{
        let p = crate::pipeline::Pipeline::new($head, $tail);
        pipelines!(@inner p $(, $rest)*)
    }};

    (@inner $acc:expr, $next:expr $(, $rest:expr)*) => {
        pipelines!(@inner $acc.pipe($next) $(, $rest)*)
    };

    (@inner $acc:expr) => {
        $acc
    };
}