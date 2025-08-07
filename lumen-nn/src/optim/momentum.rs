use lumen_core::Tensor;
use super::Optim;

pub struct Momentum {
    parameters: Vec<Tensor>,
    velocity: Vec<Tensor>,
    config: MomentumConfig,
}

pub struct MomentumConfig {
    learn_rate: f64,
    momentum: f64,
}

impl Momentum {
    pub fn new<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, learn_rate: f64, momentum: f64) -> Self {
        Self::with_config(parameters, MomentumConfig::new(learn_rate, momentum))
    }

    pub fn with_config<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, config: MomentumConfig) -> Self {
        let parameters: Vec<_> = parameters.cloned().collect();
        let velocity = parameters.iter().map(|param| Tensor::zeros_like(param)).collect();
        Self {
            parameters,
            velocity,
            config
        }
    }

    pub fn learn_rate(&self) -> f64 {
        self.config.learn_rate
    }

    pub fn momentum(&self) -> f64 {
        self.config.momentum
    }
}

impl Optim for Momentum {
    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        self.parameters.iter()
    }

    fn step(&mut self) {
        for (param, velocity) in self.parameters.iter_mut().zip(self.velocity.iter_mut()) {
            // v = momentum * v + grad
            // p += -lr * v
            let grad = param.grad().unwrap();

            for (v, &g) in velocity.iter_mut().zip(grad.iter()) {
                *v = self.config.momentum * *v + g;
            }

            for (p, &v) in param.iter_mut().zip(velocity.iter()) {
                *p += -self.config.learn_rate * v
            }
        }
    }
}

impl MomentumConfig {
    pub fn new(learn_rate: f64, momentum: f64) -> Self {
        Self { learn_rate, momentum }
    }
}

impl Default for MomentumConfig {
    /// Default: learn_rate: 0.001, momentum: 0.9
    fn default() -> Self {
        Self::new(0.001, 0.9)
    }
}