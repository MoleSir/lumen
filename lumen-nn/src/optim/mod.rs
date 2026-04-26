mod sgd;
mod adam;
mod adamw;
mod rms_prop;
mod momentum;
use std::collections::HashMap;

pub use sgd::*;
pub use adam::*;
pub use adamw::*;
pub use rms_prop::*;
pub use momentum::*;
pub mod lr_scheduler;
use lumen_core::{DynTensor, FloatDType, GradStore, Tensor};
use crate::NnResult;

pub trait Optimizer {
    type Scalar: FloatDType;

    fn get_lr(&self) -> f64;
    fn set_lr(&mut self, lr: f64);
    fn step(&mut self, grads: &GradStore<Self::Scalar>) -> NnResult<()>;
    fn named_states(&self) -> HashMap<String, Tensor<Self::Scalar>>;
    fn load_named_states(&mut self, states: &HashMap<String, DynTensor>) -> NnResult<()>;
}

pub struct LrScheduler<Opt, P> {
    pub optimizer: Opt,
    pub policy: P,
}

pub trait LrSchedulePolicy {
    fn step(&mut self, lr: f64) -> f64;
}

impl<Opt, P> LrScheduler<Opt, P> 
where 
    Opt: Optimizer,
    P: LrSchedulePolicy,
{
    pub fn new(optimizer: Opt, policy: P) -> Self {
        Self { optimizer, policy }
    }
}

impl<Opt, P> Optimizer for LrScheduler<Opt, P> 
where 
    Opt: Optimizer,
    P: LrSchedulePolicy
{
    type Scalar = Opt::Scalar;

    #[inline]
    fn get_lr(&self) -> f64 {
        self.optimizer.get_lr()
    }

    #[inline]
    fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    fn step(&mut self, grads: &GradStore<Opt::Scalar>) -> NnResult<()> {
        self.optimizer.step(grads)?;
        let lr = self.policy.step(self.optimizer.get_lr());
        self.optimizer.set_lr(lr);
        Ok(())
    }

    #[inline]
    fn named_states(&self) -> HashMap<String, Tensor<Self::Scalar>> {
        self.optimizer.named_states()
    }

    #[inline]
    fn load_named_states(&mut self, states: &HashMap<String, DynTensor>) -> NnResult<()> {
        self.optimizer.load_named_states(states)
    }
}

/*

| 简称 | 全称 (English) | 中文译名 | 核心记忆点 |
| :--- | :--- | :--- | :--- |
| SGD | Stochastic Gradient Descent | 随机梯度下降 | 每次只看一个或一小批样本。 |
| Momentum| Momentum | 动量法 | 给梯度加个“惯性”，像小球滚下坡。 |
| NAG | Nesterov Accelerated Gradient | Nesterov 加速梯度 | 往前看一步再算梯度（有先见之明）。 |
| AdaGrad | Adaptive Gradient | 自适应梯度 | 给每个参数不同的学习率，但后期太慢。 |
| Adadelta| Adaptive Delta | (无直接译名) | 改进 AdaGrad，不需要手动调学习率。 |
| RMSProp | Root Mean Square Propagation | 均方根传播 | 解决 AdaGrad 过慢问题，只看最近的波动。 |
| Adam | Adaptive Moment Estimation | 自适应矩估计 | Momentum + RMSProp 的合体。 |
| AdamW | Adam with Weight Decay | Adam 权重衰减版 | 把权重衰减从梯度计算中剥离出来。 |
| Lion | Evolved Sign Momentum | (进化符号动量) | Google 用 AI 搜出来的优化器，只看正负号。 |

*/