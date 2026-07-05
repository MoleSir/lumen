use std::collections::HashMap;

use lumen_core::{DynTensor, FloatDType, GradStore, NumDType, Tensor};
use crate::{NnError, NnResult};

use super::Optimizer;

#[derive(Clone, Debug)]
pub struct MomentumConfig<T: FloatDType> {
    pub lr: T,
    pub momentum: T,
    pub weight_decay: T,
    pub dampening: T,
    pub nesterov: bool,
}

impl<T: FloatDType> Default for MomentumConfig<T> {
    fn default() -> Self {
        Self {
            lr: T::from_f64(0.001),
            momentum: T::from_f64(0.9),     
            weight_decay: T::from_f64(0.0), 
            dampening: T::from_f64(0.0),    
            nesterov: false,     
        }
    }
}

#[derive(Debug)]
struct MomentumParam<T: FloatDType> {
    param: Tensor<T>,
    velocity: Tensor<T>,
}

#[derive(Debug)]
pub struct Momentum<T: FloatDType> {
    params: Vec<MomentumParam<T>>,
    config: MomentumConfig<T>,
}

impl<T: FloatDType> Momentum<T> {
    pub fn new(params: impl Into<Vec<Tensor<T>>>, config: MomentumConfig<T>) -> lumen_core::Result<Self> {
        let params: Vec<_> = params.into();
        let mut m_params = vec![];
        for param in params.into_iter() {
            let velocity = Tensor::zeros_like(&param)?; 
            m_params.push(MomentumParam {
                param,
                velocity,
            });
        }

        Ok(Self { params: m_params, config })
    }
}

impl<T: FloatDType> Optimizer for Momentum<T> {
    type Scalar = T;

    fn step(&mut self, grads: &GradStore<T>) -> NnResult<()> {
        let _guard = lumen_core::NoGradGuard::new();

        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;
        let dampening = self.config.dampening;
        let nesterov = self.config.nesterov;
        
        let zero = T::zero();
        let one = T::one();

        for param in self.params.iter_mut() {
            if let Some(g) = grads.get(&param.param) {
                let mut d_p = g.clone();
                
                // 权重衰减
                // 在梯度上直接加上 L2 正则化的导数。
                if weight_decay != zero {
                    d_p.add_(weight_decay * &param.param)?; 
                }
                
                // momentum: 保留多少历史 v
                // dampening: 保留多少当前梯度
                // v = v * momentum + d_p * (1 - dampening)                
                if momentum != zero {
                    if dampening != zero {
                        let scale = one - dampening;
                        d_p.mul_(scale)?; 
                    }

                    // 融合历史速度 + 当前梯度更新新的 v
                    // v = v * momentum + d_p
                    param.velocity.mul_(momentum)?;
                    param.velocity.add_(&d_p)?;
                
                    if nesterov {
                        d_p.add_(momentum * &param.velocity)?;
                    } else {
                        d_p = param.velocity.clone(); 
                    }
                }

                param.param.sub_(lr * &d_p)?;
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