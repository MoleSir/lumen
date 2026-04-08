use lumen_core::{no_grad, FloatDType, GradStore, Tensor};
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

impl<T: FloatDType> Optimizer<T> for SGD<T> {
    type Error = lumen_core::Error;
    fn step(&mut self, grads: &GradStore<T>) -> lumen_core::Result<()> {
        no_grad!();
        for var in self.params.iter() {
            if let Some(grad) = grads.get(var) {
                var.sub_(self.learning_rate * grad)?;
            }
        }
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

impl<T: FloatDType> Optimizer<T> for SGDM<T> {
    type Error = lumen_core::Error;

    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error> {
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
}