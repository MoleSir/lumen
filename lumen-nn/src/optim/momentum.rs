use lumen_core::{FloatDType, GradStore, Tensor};
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
        let mut sgd_params = vec![];
        for param in params.into_iter() {
            let velocity = Tensor::zeros_like(&param)?; 
            sgd_params.push(MomentumParam {
                param,
                velocity,
            });
        }

        Ok(Self { params: sgd_params, config })
    }
}

impl<T: FloatDType> Optimizer<T> for Momentum<T> {
    type Error = lumen_core::Error;

    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error> {
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
                
                if weight_decay != zero {
                    d_p.add_(weight_decay * &param.param)?; 
                }

                // v = v * momentum + d_p * (1 - dampening)                
                if momentum != zero {
                    if dampening != zero {
                        let scale = one - dampening;
                        // d_p = d_p * (1 - dampening)
                        d_p.mul_(scale)?; 
                    }
                    
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
}