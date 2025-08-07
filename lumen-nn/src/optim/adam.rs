use lumen_core::Tensor;
use super::Optim;

pub struct Adam {
    parameters: Vec<Tensor>,
    mv: Vec<(Tensor, Tensor)>,
    config: AdamConfig,
    time: f64,
}

pub struct AdamConfig {
    learn_rate: f64,
    beta: (f64, f64),
    epsilon: f64,
}

impl Adam {
    pub fn new<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, learn_rate: f64, beta: (f64, f64), epsilon: f64) -> Self {
        Self::with_config(parameters, AdamConfig::new(learn_rate, beta, epsilon))
    }

    pub fn with_config<'a, I: Iterator<Item = &'a Tensor>>(parameters: I, config: AdamConfig) -> Self {
        let parameters: Vec<_> = parameters.cloned().collect();
        let mv = parameters.iter()
            .map(|param| (Tensor::zeros_like(param), Tensor::zeros_like(param))).collect();
        Self { 
            parameters,
            config,
            mv,
            time: 0.,
        }
    }
    
    pub fn learn_rate(&self) -> f64 {
        self.config.learn_rate
    }

    pub fn epsilon(&self) -> f64 {
        self.config.epsilon
    }

    pub fn beta(&self) -> (f64, f64) {
        self.config.beta
    }
}

impl Optim for Adam {
    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        self.parameters.iter()
    }

    fn step(&mut self) {
        self.time += 1.;
        for (param, (cache0, cache1)) in self.parameters.iter_mut().zip(self.mv.iter_mut()) {
            let grad = param.grad().unwrap();

            for ((&g, (c0, c1)), p) in 
                grad.iter()
                    .zip(cache0.iter_mut().zip(cache1.iter_mut())) 
                    .zip(param.iter_mut())
            {
                *c0 = self.config.beta.0 * *c0 + (1. - self.config.beta.0) * g;
                *c1 = self.config.beta.1 * *c1 + (1. - self.config.beta.1) * g.powf(2.);

                let m_hat = *c0 / (1. - self.config.beta.0.powf(self.time));
                let v_hat = *c1 / (1. - self.config.beta.1.powf(self.time));

                *p += -self.config.learn_rate * m_hat / (v_hat.sqrt() + self.config.epsilon);
            }
        }
    }
}

impl AdamConfig {
    pub fn new(learn_rate: f64, beta: (f64, f64), epsilon: f64) -> Self {
        Self { learn_rate, beta, epsilon }
    }
}

impl Default for AdamConfig {
    /// Default: learn_rate: 0.001, beta: (0.9, 0.999), epsilon: 1e-8
    fn default() -> Self {
        Self::new(0.001, (0.9, 0.999), 1e-8)
    }
}