use super::{LrSchedulePolicy, LrScheduler, Optimizer};
use std::ops::{Deref, DerefMut};

// ============================================================================== //
//                                StepLR
// ============================================================================== //

#[derive(Debug, Clone, Copy)]
pub struct StepLRPolicy {
    pub step_size: usize,
    pub gamma: f64,
    pub current_step: usize,
}

impl StepLRPolicy {
    pub fn new(step_size: usize, gamma: f64) -> Self {
        Self { step_size, gamma, current_step: 0 }
    }
}

impl LrSchedulePolicy for StepLRPolicy {
    fn step(&mut self, lr: f64) -> f64 {
        self.current_step += 1;
        if self.current_step % self.step_size == 0 {
            lr * self.gamma
        } else {
            lr
        }
    }
}

pub struct StepLR<Opt>(LrScheduler<Opt, StepLRPolicy>);

impl<Opt: Optimizer> StepLR<Opt> {
    pub fn new(optimizer: Opt, step_size: usize, gamma: f64) -> Self {
        let policy = StepLRPolicy::new(step_size, gamma);
        Self(LrScheduler::new(optimizer, policy))
    }
}

impl<Opt> Deref for StepLR<Opt> {
    type Target = Opt;
    fn deref(&self) -> &Self::Target {
        &self.0.optimizer
    }
}

impl<Opt> DerefMut for StepLR<Opt> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.optimizer
    }
}

// ============================================================================== //
//                                CosineAnnealingLR
// ============================================================================== //

#[derive(Debug, Clone, Copy)]
pub struct CosineAnnealingLRPolicy {
    pub t_max: usize,
    pub eta_min: f64,
    pub initial_lr: f64,
    pub current_step: usize,
}

impl CosineAnnealingLRPolicy {
    pub fn new(initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self { t_max, eta_min, initial_lr, current_step: 0 }
    }
}

impl LrSchedulePolicy for CosineAnnealingLRPolicy {
    fn step(&mut self, _lr: f64) -> f64 {
        self.current_step += 1;
        let t_cur = self.current_step.min(self.t_max);
        
        // 余弦退火公式: eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * t_cur / t_max))
        use std::f64::consts::PI;
        self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * 
            (1.0 + (PI * t_cur as f64 / self.t_max as f64).cos())
    }
}

pub struct CosineAnnealingLR<Opt>(LrScheduler<Opt, CosineAnnealingLRPolicy>);

impl<Opt: Optimizer> CosineAnnealingLR<Opt> {
    pub fn new(optimizer: Opt, initial_lr: f64, t_max: usize, eta_min: f64) -> Self {
        let policy = CosineAnnealingLRPolicy::new(initial_lr, t_max, eta_min);
        Self(LrScheduler::new(optimizer, policy))
    }
}

impl<Opt> Deref for CosineAnnealingLR<Opt> {
    type Target = Opt;
    fn deref(&self) -> &Self::Target {
        &self.0.optimizer
    }
}

impl<Opt> DerefMut for CosineAnnealingLR<Opt> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.optimizer
    }
}