use std::collections::HashMap;

use lumen_core::{DynTensor, FloatDType, GradStore, NumDType, Tensor};
use crate::{NnError, NnResult};

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

impl<T: FloatDType> AdamWConfig<T> {
    pub fn lr(mut self, lr: T) -> Self {
        self.lr = lr;
        self
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

impl<T: FloatDType> Optimizer for AdamW<T> {
    type Scalar = T;

    fn step(&mut self, grads: &GradStore<T>) -> NnResult<()> {
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
                v.mul_(beta2)?.add_((T::one() - beta2) * g.sqr()?)?;

                let m_hat = scale_m * m;
                let v_hat =  scale_v * v;

                let adjusted_grad = m_hat / (v_hat.sqrt()? + self.config.eps);
                // 引入 l2 正则
                param.param.mul_(T::one() - lr_lambda)?;
                param.param.sub_(lr * adjusted_grad)?;
            }
        }

        Ok(())
    }

    fn get_lr(&self) -> f64 {
        <T as NumDType>::to_f64(self.config.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        self.config.lr = T::from_f64(lr);
    }

    fn named_states(&self) -> HashMap<String, Tensor<Self::Scalar>> {
        let mut tensors = HashMap::new();
        for (i, param) in self.params.iter().enumerate() {
            tensors.insert(first_moment_name(i), param.first_moment.clone());
            tensors.insert(second_moment_name(i), param.second_moment.clone()); 
        }
        tensors
    }

    fn load_named_states(&mut self, states: &HashMap<String, DynTensor>) -> NnResult<()> {
        for (i, param) in self.params.iter().enumerate() {
            let first_name = first_moment_name(i);
            let first_moment = states.get(&first_name).ok_or_else(|| NnError::ParamNotFound(first_name, "load_named_states"))?;
            let first_moment = first_moment.as_tensor::<T>()?;
            param.first_moment.copy_(&first_moment)?;

            let second_name = second_moment_name(i);
            let second_moment = states.get(&second_name).ok_or_else(|| NnError::ParamNotFound(second_name, "load_named_states"))?;
            let second_moment = second_moment.as_tensor::<T>()?;
            param.second_moment.copy_(&second_moment)?;
        }
        Ok(())
    }
}

fn first_moment_name(i: usize) -> String {
    format!("first_moment_{}", i)
}

fn second_moment_name(i: usize) -> String {
    format!("second_moment_{}", i)
}