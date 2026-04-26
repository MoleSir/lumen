use std::collections::HashMap;

use lumen_core::{no_grad, DynTensor, FloatDType, GradStore, NumDType, Tensor};
use crate::{NnError, NnResult};

use super::Optimizer;

//======================================================================//
//               SGD
//======================================================================//

#[derive(Debug)]
pub struct SGD<T: FloatDType> {
    pub params: Vec<Tensor<T>>,
    pub learning_rate: T,
}

impl<T: FloatDType> SGD<T> {
    pub fn new(params: impl Into<Vec<Tensor<T>>>, learning_rate: T) -> Self {
        Self { params: params.into(), learning_rate }
    }
}

impl<T: FloatDType> Optimizer for SGD<T> {
    type Scalar = T;

    fn step(&mut self, grads: &GradStore<T>) -> NnResult<()> {
        no_grad!();
        for var in self.params.iter() {
            if let Some(grad) = grads.get(var) {
                var.sub_(self.learning_rate * grad)?;
            }
        }
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        <T as NumDType>::to_f64(self.learning_rate)
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = T::from_f64(lr);
    }

    #[inline]
    fn named_states(&self) -> HashMap<String, Tensor<Self::Scalar>> {
        HashMap::new()
    }

    #[inline]
    fn load_named_states(&mut self, _states: &HashMap<String, DynTensor>) -> NnResult<()> {
        Ok(())
    }
}

//======================================================================//
//               SGD-M
//======================================================================//

pub struct SGDM<T: FloatDType> {
    pub params: Vec<SGDMParam<T>>,
    pub learning_rate: T,
    pub momentum: T,
    pub nesterov: bool,
}

pub struct SGDMParam<T: FloatDType> {
    pub param: Tensor<T>,
    pub velocity: Tensor<T>,
}

impl<T: FloatDType> Optimizer for SGDM<T> {
    type Scalar = T;

    fn step(&mut self, grads: &GradStore<T>) -> Result<(), NnError> {
        no_grad!();
        for SGDMParam { param, velocity } in self.params.iter_mut() {
            if let Some(grad) = grads.get(param) {
                // 1. 更新动量：v = momentum * v + grad
                velocity.mul_(self.momentum)?;
                velocity.add_(grad)?;
                
                let final_grad = if self.nesterov {
                    // grad + momentum * velocity
                    grad + self.momentum * velocity.clone()
                } else {
                    velocity.clone()
                };

                param.sub(self.learning_rate * final_grad)?;
            }
        }

        Ok(())
    }

    fn get_lr(&self) -> f64 {
        <T as NumDType>::to_f64(self.learning_rate)
    }

    fn set_lr(&mut self, lr: f64) {
        self.learning_rate = T::from_f64(lr);
    }

    fn named_states(&self) -> HashMap<String, Tensor<Self::Scalar>> {
        self.params.iter()
            .enumerate()
            .map(|(i, param)| (format!("{}", i), param.velocity.clone()))
            .collect()
    }

    fn load_named_states(&mut self, states: &HashMap<String, DynTensor>) -> NnResult<()> {
        for (i, param) in self.params.iter().enumerate() {
            let velocity = &param.velocity;
            let key: String = format!("{}", i);
            let src = states.get(&key).ok_or_else(|| NnError::ParamNotFound(key.clone(), "load_named_states"))?;
            let src = src.as_tensor::<T>()?;
            velocity.copy_(&src)?;
        }
        Ok(())
    }
}