use anyhow::Context;
use hands_on_rl::env::CartPoleEnv;
use lumen_core::{FloatDType, IndexOp, NumDType, Tensor, D};
use lumen_nn::{optim::{Adam, AdamConfig, Optimizer}, Linear, Module};
use plotters::{chart::ChartBuilder, prelude::{BitMapBackend, IntoDrawingArea}, series::LineSeries, style::{BLUE, WHITE}};
use rand::RngExt;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    pub const LR: f64 = 5e-4;
    pub const NUM_EPISODES: usize = 100;
    pub const HIDDEN_DIM: usize = 128;
    pub const GAMMA: f64 = 0.98;

    let mut env = CartPoleEnv::new();

    let mut agent = Reinforce::new(4, HIDDEN_DIM, 2, LR, GAMMA)?;
    let mut return_list = vec![];
    for i in 0..10 {
        for e in 0..NUM_EPISODES {
            let mut episode_return = 0.;
            let mut state = env.reset();
            let mut done = false;
            
            let mut episode_data = vec![];
            
            while !done {
                let action = agent.take_action(&state)?;
                let (next_state, reward, is_done) = env.step(action);
                
                let state_tensor = Tensor::new(state.clone())?;
                episode_data.push((state_tensor, action, reward));
    
                state = next_state;
                episode_return += reward;
                done = is_done;
            }
    
            agent.update(episode_data).context("update")?;
            
            println!("Trun {}, Episode: {}, Return: {}", i, e, episode_return);
            return_list.push((return_list.len(), episode_return));
        }
    }

    println!("render pic");
    println!("{}", return_list.len());
    let root = BitMapBackend::new(&"./result/ch09/returns.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return RF", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max.min(10000.0))?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    println!("done");
    Ok(())
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

    /// 输入状态，输出每个 action 的概率
    /// 
    /// ## Args
    /// -state: (..., state_dim)
    /// 
    /// ## Return
    /// - probs: (..., action_dim)
    pub fn forward(&self, state: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // (..., state_dim) => (..., hidden_dim)
        let h = self.fc1.forward(state)?.relu()?;
        let action = self.fc2.forward(&h)?;
        let probs = lumen_nn::functional::softmax(&action, D::Minus1)?;
        Ok(probs)
    }
}

pub struct Reinforce<T: FloatDType> {
    pub state_dim: usize,
    pub hidden_dim: usize,
    pub action_dim: usize,

    pub policy_net: PolicyNet<T>,
    pub optimizer: Adam<T>,

    pub gamma: f64,
}

impl<T: FloatDType> Reinforce<T> {
    pub fn new(
        state_dim: usize, hidden_dim: usize, action_dim: usize,
        learning_rate: T, gamma: f64, 
    ) -> anyhow::Result<Self> {
        let policy_net = PolicyNet::new(state_dim, hidden_dim, action_dim)?;
        let mut optimizer_config= AdamConfig::default();
        optimizer_config.lr = learning_rate;
        let optimizer = Adam::new(policy_net.params(), optimizer_config)?;

        Ok(Self {
            state_dim,
            hidden_dim,
            action_dim,

            policy_net,
            optimizer,

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

    pub fn update(&mut self, transition_dict: Vec<(Tensor<T>, usize, f64)>) -> anyhow::Result<()> {
        let mut g = T::zero();
        let mut returns = vec![];
    
        // --- 步骤 1: 计算每个时刻的累计折扣回报 G_t ---
        for (_, _, reward) in transition_dict.iter().rev() {
            g = T::from_f64(self.gamma) * g + T::from_f64(*reward);
            returns.push(g);
        }
        returns.reverse(); // 恢复到正向顺序 [G_1, G_2, ..., G_T]
    
        // --- 步骤 2: 回报归一化 (Reward Normalization) ---
        // 这是稳定训练的关键！将 G_t 减去均值并除以标准差
        let returns_f64: Vec<f64> = returns.iter().map(|v| <T as NumDType>::to_f64(*v)).collect();
        let mean = returns_f64.iter().sum::<f64>() / returns_f64.len() as f64;
        let var = returns_f64.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / returns_f64.len() as f64;
        let std = (var + 1e-8).sqrt();
    
        // --- 步骤 3: 累积整个回合的损失 ---
        let mut total_loss = None;
    
        for (i, (state, action, _)) in transition_dict.into_iter().enumerate() {
            let norm_g = T::from_f64((returns_f64[i] - mean) / std);
    
            let probs = self.policy_net.forward(&state.unsqueeze(0)?)?.squeeze(0)?;
            let prob = probs.index(action)?;
            
            // 使用 ln(prob + eps) 防止数值崩溃
            let log_prob = (prob + T::from_f64(1e-9)).ln()?;
            
            // REINFORCE 损失函数: -log_prob * G_t
            let loss = log_prob.neg()? * norm_g;
    
            if let Some(tl) = total_loss {
                total_loss = Some(tl + loss);
            } else {
                total_loss = Some(loss);
            }
        }
    
        // --- 步骤 4: 统一更新一次参数 ---
        if let Some(loss) = total_loss {
            // 如果你的框架支持，可以取平均 loss：loss / episode_length
            let grads = loss.backward()?;
            self.optimizer.step(&grads)?;
        }
    
        Ok(())
    }
}