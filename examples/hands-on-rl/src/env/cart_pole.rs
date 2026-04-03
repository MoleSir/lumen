use gym_rs::{core::Env, envs::classical_control::cartpole::CartPoleEnv as GymCartPoleEnv, utils::renderer::RenderMode};

pub struct CartPoleEnv(GymCartPoleEnv);

pub type State = Vec<f64>;
pub type Action = usize;

impl CartPoleEnv {
    pub fn new() -> Self {
        Self(GymCartPoleEnv::new(RenderMode::None))
    }

    /// reset env and return init state
    pub fn reset(&mut self) -> State {
        let (obs, _) = self.0.reset(Some(25), false, None);
        obs.into()
    }

    /// execute action, return next_state, reward and done 
    pub fn step(&mut self, action: Action) -> (State, f64, bool) {
        let result = self.0.step(action);
        let next_state: Vec<f64> = result.observation.into();
        let reward = result.reward.into();
        (next_state, reward, result.done)
    }
}