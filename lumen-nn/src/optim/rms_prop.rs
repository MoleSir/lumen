use lumen_core::{FloatDType, GradStore, Tensor};
use super::Optimizer;

#[derive(Clone, Debug)]
pub struct RMSPropConfig<T: FloatDType> {
    pub lr: T,
    pub alpha: T,        // 平滑常数（通常设为 0.99）
    pub eps: T,          // 极小值（通常设为 1e-8）
    pub weight_decay: T,
}

impl<T: FloatDType> Default for RMSPropConfig<T> {
    fn default() -> Self {
        Self {
            lr: T::from_f64(0.01),
            alpha: T::from_f64(0.99),
            eps: T::from_f64(1e-8),
            weight_decay: T::from_f64(0.0),
        }
    }
}

#[derive(Debug)]
struct RMSPropParam<T: FloatDType> {
    param: Tensor<T>,
    square_avg: Tensor<T>, 
}

#[derive(Debug)]
pub struct RMSProp<T: FloatDType> {
    params: Vec<RMSPropParam<T>>,
    config: RMSPropConfig<T>,
}

impl<T: FloatDType> RMSProp<T> {
    pub fn new(params: impl Into<Vec<Tensor<T>>>, config: RMSPropConfig<T>) -> lumen_core::Result<Self> {
        let params: Vec<_> = params.into();
        let mut m_params = vec![];
        for param in params.into_iter() {
            let square_avg = Tensor::zeros_like(&param)?;
            m_params.push(RMSPropParam {param, square_avg,});
        }

        Ok(Self { params: m_params, config })
    }
}

impl<T: FloatDType> Optimizer<T> for RMSProp<T> {
    type Error = lumen_core::Error;

    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error> {
        let _guard = lumen_core::NoGradGuard::new();

        let lr = self.config.lr;
        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let weight_decay = self.config.weight_decay;

        let zero = T::zero();
        let one = T::one();

        for p in self.params.iter_mut() {
            if let Some(g) = grads.get(&p.param) {
                let d_p = g.clone();

                // 1. Weight Decay
                if weight_decay != zero {
                    d_p.add_(weight_decay * &p.param)?;
                }

                // 2. 更新平方平均: v = v * alpha + (1 - alpha) * g^2
                p.square_avg.mul_(alpha)?;
                p.square_avg.add_((one - alpha) * d_p.sqr()?)?;

                // 3. 计算分母: denom = sqrt(square_avg) + eps
                let denom = p.square_avg.sqrt()?.add(eps)?;

                // 4. 标准 RMSProp 更新: param = param - lr * (d_p / denom)
                p.param.sub_(lr * d_p.div(&denom)?)?;
            }
        }

        Ok(())
    }
}