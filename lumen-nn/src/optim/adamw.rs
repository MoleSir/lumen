use lumen_core::{FloatDType, GradStore, Tensor};
use super::Optimizer;

#[derive(Clone, Debug)]
pub struct AdamWConfig<T: FloatDType> {
    pub lr: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub weight_decay: T,
}

impl<T: FloatDType> Default for AdamWConfig<T> {
    fn default() -> Self {
        Self {
            lr: T::from_f64(0.001),
            beta1: T::from_f64(0.9),
            beta2: T::from_f64(0.999),
            eps: T::from_f64(1e-8),
            weight_decay: T::from_f64(0.01),
        }
    }
}

#[derive(Debug)]
struct AdamWParam<T: FloatDType> {
    param: Tensor<T>,
    first_moment: Tensor<T>,
    second_moment: Tensor<T>,
}

#[derive(Debug)]
pub struct AdamW<T: FloatDType> {
    params: Vec<AdamWParam<T>>,
    step_t: usize,
    pub config: AdamWConfig<T>,
}

impl<T: FloatDType> AdamW<T> {
    pub fn new(params: impl Into<Vec<Tensor<T>>>, config: AdamWConfig<T>) -> lumen_core::Result<Self> {
        let params: Vec<_> = params.into();
        let mut adamw_params = vec![];
        for param in params.into_iter() {
            let first_moment = Tensor::zeros_like(&param)?; // no need grad
            let second_moment = Tensor::zeros_like(&param)?; // no need grad
            adamw_params.push(AdamWParam {
                param, first_moment, second_moment
            });
        }

        Ok(Self { params: adamw_params, step_t: 0, config })
    }
}

impl<T: FloatDType> Optimizer<T> for AdamW<T> {
    type Error = lumen_core::Error;
    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error> {
        let _guard = lumen_core::NoGradGuard::new();

        self.step_t += 1;
        let lr = self.config.lr;
        let lambda = self.config.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let scale_m = T::one() / (T::one() - beta1.powi(self.step_t as i32));
        let scale_v = T::one() / (T::one() - beta2.powi(self.step_t as i32));
        
        for param in self.params.iter_mut() {
            let m = &param.first_moment;
            let v = &param.second_moment;
            if let Some(g) = grads.get(&param.param) {
                m.mul_(beta1)?.add_((T::one() - beta1) * g)?;
                v.mul_(beta2)?.add_((T::one() - beta2) * g.sqr())?;

                let m_hat = scale_m * m;
                let v_hat =  scale_v * v;

                let adjusted_grad = m_hat / (v_hat.sqrt() + self.config.eps);
                param.param.mul_(T::one() - lr_lambda)?;
                param.param.sub_(lr * adjusted_grad)?;
            }
        }

        Ok(())
    }
}