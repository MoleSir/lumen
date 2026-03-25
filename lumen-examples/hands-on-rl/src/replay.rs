use lumen_core::{FloatDType, Tensor};
use rand::seq::IndexedRandom;

pub struct Transition<T: FloatDType> {
    pub state: Vec<T>,
    pub action: usize,
    pub reward: T,
    pub next_state: Vec<T>,
    pub done: bool,
}

pub struct ReplayBuffer<T: FloatDType> {
    capacity: usize,
    buffer: Vec<Transition<T>>,
    position: usize,
}

impl<T: FloatDType> ReplayBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, buffer: Vec::with_capacity(capacity), position: 0 }
    }

    pub fn push(&mut self, transition: Transition<T>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    pub fn sample_transition(&self, batch_size: usize) -> Vec<&Transition<T>> {
        let mut rng = rand::rng();
        self.buffer.sample(&mut rng, batch_size).collect()
    }

    pub fn sample(&self, batch_size: usize) -> anyhow::Result<(Tensor<T>, Vec<usize>, Vec<T>, Tensor<T>, Vec<bool>)> {

        let mut states = vec![];
        let mut actions = vec![];
        let mut rewards = vec![];
        let mut next_states = vec![];
        let mut dones = vec![];

        for sample in self.sample_transition(batch_size) {
            states.extend(sample.state.iter().cloned());
            actions.push(sample.action);
            rewards.push(sample.reward);
            next_states.extend(sample.next_state.iter().cloned());
            dones.push(sample.done);
        }

        let state_dim = states.len() / batch_size;
        let states = Tensor::from_vec(states, (batch_size, state_dim))?;
        let next_states = Tensor::from_vec(next_states, (batch_size, state_dim))?;

        Ok((states, actions, rewards, next_states, dones))
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}