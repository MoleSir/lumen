use lumen_core::Tensor;
use super::Optim;

pub struct SDG {
    parameters: Vec<Tensor>,
    config: SDGConfig,
}

pub struct SDGConfig {
    learn_rate: f64,
}

impl SDG {
    pub fn new<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, learn_rate: f64) -> Self {
        Self { 
            parameters: parameters.cloned().collect(),
            config: SDGConfig::new(learn_rate),
        }
    }

    pub fn with_config<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, config: SDGConfig) -> Self {
        Self { 
            parameters: parameters.cloned().collect(),
            config,
        }
    }

    pub fn learn_rate(&self) -> f64 {
        self.config.learn_rate
    }
}

impl Optim for SDG {
    fn step(&mut self) {
        for param in self.parameters.iter() {
            let grad = param.grad().unwrap();
            for (v, &g) in param.iter_mut().zip(grad.iter()) {
                *v -= self.learn_rate() * g;
            }
        }
    }

    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        self.parameters.iter()
    }
}

impl SDGConfig {
    pub fn new(learn_rate: f64) -> Self {
        Self { learn_rate }
    }

    pub fn learn_rate(self, learn_rate: f64) -> Self {
        Self { learn_rate }
    }
}

impl Default for SDGConfig {
    /// Default: learn_rate: 0.001
    fn default() -> Self {
        Self::new(0.001)
    }
}