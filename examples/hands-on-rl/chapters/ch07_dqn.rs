use core::f64;
use anyhow::Context;
use hands_on_rl::{env::CartPoleEnv, replay::{ReplayBuffer, Transition}, utils};
use lumen_core::{FloatDType, Tensor};
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
    pub const EPSILON: f64 = 0.01;
    pub const TARGET_UPDATE: usize = 100;
    pub const BUFFER_SIZE: usize = 10_000;
    pub const MINIMAL_SIZE: usize = 500;
    pub const BATCH_SIZE: usize = 64;

    let mut env = CartPoleEnv::new();
    let mut replay_buffer = ReplayBuffer::<f64>::new(BUFFER_SIZE);

    let mut agent = DQN::new(4, HIDDEN_DIM, 2, LR, GAMMA, EPSILON, TARGET_UPDATE)?;

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
    let root = BitMapBackend::new(&"./result/ch07/returns.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return DQN", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    println!("done");
    Ok(())
}

// ============================================================== //
//                 DQN
// ============================================================== //

#[derive(Module, Clone)]
pub struct QNet<T: FloatDType> {
    pub fc1: Linear<T>,
    pub fc2: Linear<T>,
}

impl<T: FloatDType> QNet<T> {
    pub fn new(state_dim: usize, hidden_dim: usize, action_dim: usize) -> anyhow::Result<Self> {
        let fc1 = Linear::new(state_dim, hidden_dim, true, None)?;
        let fc2 = Linear::new(hidden_dim, action_dim, true, None)?;
        Ok(Self { fc1, fc2 })
    }

    /// 输入状态，输出每个 action 的分数
    /// 
    /// ## Args
    /// -state: (..., state_dim)
    /// 
    /// ## Return
    /// - actions: (..., action_dim)
    pub fn forward(&self, state: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // (..., state_dim) => (..., hidden_dim)
        let h = self.fc1.forward(state)?.relu()?;
        let action = self.fc2.forward(&h)?;
        Ok(action)
    }
}

pub struct DQN<T: FloatDType> {
    pub state_dim: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,

    pub q_net: QNet<T>,
    pub target_q_net: QNet<T>,
    pub optimizer: Adam<T>,

    pub gamma: f64,
    pub epsilon: f64,
    pub target_update: usize,
    pub count: usize,
}

impl<T: FloatDType> DQN<T> {
    pub fn new(
        state_dim: usize, hidden_dim: usize, action_dim: usize,
        learning_rate: T, gamma: f64, epsilon: f64, target_update: usize,
    ) -> anyhow::Result<Self> {
        let q_net = QNet::new(state_dim, hidden_dim, action_dim)?;
        let target_q_net = QNet::new(state_dim, hidden_dim, action_dim)?;
        let mut optimizer_config= AdamConfig::default();
        optimizer_config.lr = learning_rate;
        let optimizer = Adam::new(q_net.params(), optimizer_config)?;

        Ok(Self {
            state_dim,
            hidden_dim,
            action_dim,

            q_net,
            target_q_net,
            optimizer,

            gamma,
            epsilon,
            target_update,
            count: 0,
        })
    }

    /// Epsilon-greedy 选择动作
    pub fn take_action(&mut self, state: &[T]) -> anyhow::Result<usize> {
        if utils::epsilon_policy(self.epsilon) {
            // self.epsilon *= 0.995;
            Ok(rand::rng().random_range(0..self.action_dim))
        } else {
            // 利用网络输出的 Q 值
            let state = Tensor::new(state.to_vec())?.unsqueeze(0)?;
            let q_values = self.q_net.forward(&state)?.flatten_all()?;
            let action = q_values.argmax(0)?.to_scalar()? as usize;
            // println!("{}", action);
            // let safe_action = action % self.action_dim; 
            Ok(action)
        }
    }
    
    pub fn update(&mut self, buffer: &ReplayBuffer<T>, batch_size: usize) -> anyhow::Result<()> {
        if buffer.len() < batch_size {
            return Ok(());
        }

        /*
            - states: (batch_size, state_dim)
            - actions: (batch_size)
            - rewards: (batch_size)
            - next_states: (batch_size, state_dim)
            - dones: (batch_size)
        */
        let (states, actions, rewards, next_states, dones) = buffer.sample(batch_size)?;
        let actions = Tensor::new(actions.into_iter().map(|v| v as u32).collect::<Vec<_>>())?;
        let rewards = Tensor::new(rewards)?; // (batch_size,)
        let mask = Tensor::new(dones.into_iter().map(|v| if v {T::zero()} else {T::one()}).collect::<Vec<_>>())?;

        // 计算当前 Q 值: Q(s, a)
        // 每个 batch 输入状态，输出每个 action 的分数
        let q_values = self.q_net.forward(&states)?; // (batch_size, action_dim)
        // 根据实际选择 actions 得到特定的那个 action 的分数
        let current_q = q_values.gather(&actions.unsqueeze(1)?, 1)?.squeeze(1)?; // (batch_size,)

        // 计算目标 Q 值: max Q_target(s', a')
        // 输入下一个状态，输出下一个状态下，输出每个 action 的分数
        let next_q_values = self.target_q_net.forward(&next_states)?; // (batch_size, action_dim)
        // 没有实际选择，所以直接选择最大 action
        let max_next_q = next_q_values.max(1)?; // (batch_size)

        // 贝尔曼方程: target = reward + gamma * max_next_q * (1 - done)
        /*
        
            更新参数：

                Q(s, a) <- Q(s, a) + \alpha * (G - Q(s, a))

            ->  Q(s, a) <- Q(s, a) + \alpha * (r + \gamma Q_{max}(S', a') - Q(s, a))

            右侧要逼近左侧，就是更新部分：(r + \gamma Q_{max}(S', a') - Q(s, a)) 尽可能小，即：

            r + \gamma Q_{max}(S', a') == Q(s, a)

            上面计算的：
            - current_q 是网络预测的 Q(s, a) 被选择特定 action 后的 q 值
            - max_next_q 是下一个状态输出的 Q(s, a) 的最大 q 值
        */
        let target_q = &rewards + T::from_f64(self.gamma) * &max_next_q * mask; // (batch_size)
        let target_q = target_q.detach();

        let loss = lumen_nn::functional::mse_loss(&current_q, &target_q, LossReduction::Mean)?;

        let grads = loss.backward()?;
        self.optimizer.step(&grads)?;

        // 更新 target
        self.count += 1;
        if self.count % self.target_update == 0 {
            // self.target_q_net = self.q_net.copy()?;
            self.target_q_net.load_named_states(&self.q_net.named_dyn_states(), true)?;
        }
        

        Ok(())
    }
}

