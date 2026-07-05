use anyhow::Context;
use hands_on_rl::{env::CartPoleEnv, replay::{ReplayBuffer, Transition}};
use lumen_core::{FloatDType, Tensor, D};
use lumen_nn::{functional::LossReduction, optim::{Adam, AdamConfig, Optimizer}, Linear, Module};
use plotters::{chart::ChartBuilder, prelude::{BitMapBackend, IntoDrawingArea}, series::LineSeries, style::{BLUE, WHITE}};
use rand::RngExt;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}


fn result_main() -> anyhow::Result<()> {
    pub const LR: f64 = 1e-3;
    pub const NUM_EPISODES: usize = 500;
    pub const HIDDEN_DIM: usize = 128;
    pub const GAMMA: f64 = 0.98;
    pub const BUFFER_SIZE: usize = 10_000;
    pub const MINIMAL_SIZE: usize = 10;
    pub const BATCH_SIZE: usize = 64;

    let mut env = CartPoleEnv::new();
    let mut replay_buffer = ReplayBuffer::<f64>::new(BUFFER_SIZE);

    let mut agent = ActorCritic::new(4, HIDDEN_DIM, 2, LR, GAMMA)?;

    let mut return_list = vec![];
    for i in 0..10 {
        for e in 0..NUM_EPISODES / 10 {
            let mut episode_return = 0.;
            let mut state = env.reset();

            let mut done = false;
            
            while !done {
                let action = agent.take_action(&state).context("take action")?;
                let (next_state, reward, is_done) = env.step(action);
                done = is_done;

                replay_buffer.push(Transition {
                    state,
                    action,
                    reward,
                    next_state: next_state.clone(),
                    done,
                });

                state = next_state;
                episode_return += reward;

                if replay_buffer.len() > MINIMAL_SIZE {
                    agent.update(&replay_buffer, BATCH_SIZE)?;
                }
            }

            println!("{},{},{}", i, e, episode_return);

            return_list.push((return_list.len(), episode_return));
        }
    }

    println!("render pic");
    println!("{}", return_list.len());
    let root = BitMapBackend::new(&"./result/ch10/returns.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return AC", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    println!("done");
    Ok(())
}

pub struct ActorCritic<T: FloatDType> {
    pub policy_net: PolicyNet<T>,
    pub policy_optimizer: Adam<T>,
    pub value_net: ValueNet<T>,
    pub value_optimizer: Adam<T>,
    pub gamma: f64,
}

impl<T: FloatDType> ActorCritic<T> {
    pub fn new(
        state_dim: usize, hidden_dim: usize, action_dim: usize,
        learning_rate: T, gamma: f64, 
    ) -> anyhow::Result<Self> {
        let mut optimizer_config= AdamConfig::default();
        optimizer_config.lr = learning_rate;

        let policy_net = PolicyNet::new(state_dim, hidden_dim, action_dim)?;
        let policy_optimizer = Adam::new(policy_net.params(), optimizer_config.clone())?;

        let value_net = ValueNet::new(state_dim, hidden_dim)?;
        let value_optimizer = Adam::new(value_net.params(), optimizer_config)?;

        Ok(Self {
            policy_net, policy_optimizer,
            value_net, value_optimizer,
            gamma
        })
    }

    pub fn take_action(&self, state: &[T]) -> anyhow::Result<usize> {
        let state = Tensor::new(state.to_vec())?.unsqueeze(0)?;
        let probs = self.policy_net
            .forward(&state)?
            .to_vec()?;
    
        let mut acc = T::zero();
        let rval = T::from_f64(rand::rng().random_range(0.0..1.0));
    
        for (i, prob) in probs.into_iter().enumerate() {
            acc += prob;
            if acc > rval {
                return Ok(i);
            }
        }
        
        Ok(0)
    }

    // pub fn update(&mut self, buffer: &ReplayBuffer<T>, batch_size: usize) -> anyhow::Result<()> {
    //     if buffer.len() < batch_size {
    //         return Ok(());
    //     }

    //     /*
    //         - states: (batch_size, state_dim)
    //         - actions: (batch_size)
    //         - rewards: (batch_size)
    //         - next_states: (batch_size, state_dim)
    //         - dones: (batch_size)
    //     */
    //     let (states, actions, rewards, next_states, dones) = buffer.sample(batch_size)?;
    //     let actions = Tensor::new(actions.into_iter().map(|v| v as u32).collect::<Vec<_>>())?.unsqueeze(1)?; // (batch_size, 1)
    //     let rewards = Tensor::new(rewards)?.unsqueeze(1)?; // (batch_size, 1)
    //     let mask = Tensor::new(dones.into_iter().map(|v| if v {T::zero()} else {T::one()}).collect::<Vec<_>>())?.unsqueeze(1)?; // (batch_size, 1)

    //     /*
    //         时序差分目标：当前奖励 + \gamma * 下一个状态的预估价值
    //         r + \gamma * V_{\omega}(s_{t+1})
    //     */
    //     let next_state_value = self.value_net
    //         .forward(&next_states)
    //         .context("value forward")?; // (batch, 1)
    //     let td_target = &rewards + T::from_f64(self.gamma) * &next_state_value * mask;
    //     let td_target = td_target.detach();

    //     /*
    //         时序差分误差：预期奖励 - 在当前状态预估的价值
    //         [r + \gamma * V_{\omega}(s_{t+1})] - V_{\omega}(s_{t})
    //     */
    //     let state_value = self.value_net
    //         .forward(&states)
    //         .context("value forward")?; // (batch, 1)
    //     let td_delta = &td_target - &state_value;
    //     // 预期和“实际”奖励的差距
    //     let td_delta = td_delta.detach();

    //     /*
    //         每个样本在 batch 中，按照选择的动作对应的 Q 值：(batch, 1)，然后取 log
    //         \log \pi_{\theta}(a_t|s_t)
    //     */
    //     let probs = self.policy_net
    //         .forward(&states)
    //         .context("policy forward")?; // (batch, action_dim)
    //     let probs = probs.gather(actions, 1).context("gather")?; // (batch, 1)
    //     let log_probs = probs.ln()?; // (batch, 1)
    //     let actor_loss = (log_probs.neg()? * td_delta).mean(0)?;

    //     /*
    //         均方误差损失函数
    //         L(\omega) = \frac{1}{2}(r + \gamma V_{\omega}(s_{t+1}) - V_{\omega}(s_t))^2
    //     */
    //     let critic_loss = lumen_nn::functional::mse_loss(&state_value, &td_target, LossReduction::Mean)?;

    //     // 更新参数
    //     let grads = actor_loss.backward()?;
    //     self.policy_optimizer.step(&grads)?;

    //     let grads = critic_loss.backward()?;
    //     self.value_optimizer.step(&grads)?;

    //     Ok(())
    // }

    pub fn update(&mut self, buffer: &ReplayBuffer<T>, batch_size: usize) -> anyhow::Result<()> {
        if buffer.len() < batch_size {
            return Ok(());
        }
    
        let (states, actions, rewards, next_states, dones) = buffer.sample(batch_size)?;
        let actions = Tensor::new(actions.into_iter().map(|v| v as u32).collect::<Vec<_>>())?.unsqueeze(1)?;
        let rewards = Tensor::new(rewards)?.unsqueeze(1)?;
        let mask = Tensor::new(dones.into_iter().map(|v| if v {T::zero()} else {T::one()}).collect::<Vec<_>>())?.unsqueeze(1)?;
    
        // 1. 计算 TD Target (修正后的加法逻辑)
        let next_state_value = self.value_net.forward(&next_states)?; 
        let td_target = &rewards + T::from_f64(self.gamma) * &next_state_value * mask;
        let td_target = td_target.detach(); // 目标不参与 Critic 梯度计算
    
        // 2. 计算 TD Delta (Advantage)
        let state_value = self.value_net.forward(&states)?;
        let td_delta = &td_target - &state_value;
        let actor_td_delta = td_delta.detach();
    
        // 3. Actor Loss
        let probs = self.policy_net.forward(&states)?;
        let chosen_probs = probs.gather(actions, 1)?;
        let log_probs = (chosen_probs + T::from_f64(1e-8)).ln()?;
        let actor_loss = (log_probs.neg()? * actor_td_delta).mean(0)?;
    
        // 4. Critic Loss (均方误差)
        let critic_loss = lumen_nn::functional::mse_loss(&state_value, &td_target, LossReduction::Mean)?;
    
        // 5. 分别更新
        let p_grads = actor_loss.backward()?;
        self.policy_optimizer.step(&p_grads)?;
    
        let v_grads = critic_loss.backward()?;
        self.value_optimizer.step(&v_grads)?;
    
        Ok(())
    }
}


#[derive(Module, Clone)]
pub struct PolicyNet<T: FloatDType> {
    pub fc1: Linear<T>,
    pub fc2: Linear<T>,
}

impl<T: FloatDType> PolicyNet<T> {
    pub fn new(state_dim: usize, hidden_dim: usize, action_dim: usize) -> anyhow::Result<Self> {
        let fc1 = Linear::new(state_dim, hidden_dim, true, None)?;
        let fc2 = Linear::new(hidden_dim, action_dim, true, None)?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, state: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // (..., state_dim) => (..., hidden_dim) => (..., action_dim)
        let h = self.fc1.forward(state)?.relu()?;
        let action = self.fc2.forward(&h)?;
        let probs = lumen_nn::functional::softmax(&action, D::Minus1)?;
        Ok(probs)
    }
}

#[derive(Module, Clone)]
pub struct ValueNet<T: FloatDType> {
    pub fc1: Linear<T>,
    pub fc2: Linear<T>,
}

impl<T: FloatDType> ValueNet<T> {
    pub fn new(state_dim: usize, hidden_dim: usize) -> anyhow::Result<Self> {
        let fc1 = Linear::new(state_dim, hidden_dim, true, None)?;
        let fc2 = Linear::new(hidden_dim, 1, true, None)?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, state: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // (..., state_dim) => (..., hidden_dim) => (..., 1)
        let h = self.fc1.forward(state)?.relu()?;
        let action = self.fc2.forward(&h)?;
        Ok(action)
    }
}