mod sgd;
mod adam;
mod adamw;
mod rms_prop;
mod momentum;
pub use sgd::*;
pub use adam::*;
pub use adamw::*;
pub use rms_prop::*;
pub use momentum::*;

use lumen_core::{FloatDType, GradStore};

pub trait Optimizer<T: FloatDType> {
    type Error: std::error::Error + Sync + Send + 'static;
    fn step(&mut self, grads: &GradStore<T>) -> Result<(), Self::Error>;
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