use std::collections::{HashMap, VecDeque};
use hands_on_rl::utils;
use plotters::{chart::ChartBuilder, prelude::{BitMapBackend, IntoDrawingArea}, series::LineSeries, style::{BLUE, WHITE}};
use rand::RngExt;


fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    test_sarsa()?;
    test_nstep_sarsa()?;
    test_qlearning()?;
    Ok(())
}

// ==================================================================== //
//                          CliffWalkingEnv
// ==================================================================== //

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl TryFrom<i32> for Action {
    type Error = anyhow::Error;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Up),
            1 => Ok(Self::Down),
            2 => Ok(Self::Left),
            3 => Ok(Self::Right),
            _ => Err(anyhow::anyhow!("invalid value {}", value))
        }
    }
}

pub type Coord = (i32, i32);
pub type State = (i32, i32);

impl Action {
    pub fn delta_pos(&self) -> Coord {
        match self {
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }
}

/// 悬崖漫步是一个非常经典的强化学习环境，它要求一个智能体从起点出发，避开悬崖行走，最终到达目标位置。
/// 有一个 m×n 的网格世界，每一个网格表示一个状态。
/// 智能体的起点是左下角的状态，目标是右下角的状态，智能体在每一个状态都可以采取 4 种动作：上、下、左、右。
/// 如果智能体采取动作后触碰到边界墙壁则状态不发生改变，否则就会相应到达下一个状态。
/// 环境中有一段悬崖，智能体掉入悬崖或到达目标状态都会结束动作并回到起点，也就是说掉入悬崖或者达到目标状态是终止状态。
/// 智能体每走一步的奖励是 −1，掉入悬崖的奖励是 −100。
///
/// (0,0)    x --->     ncol
///      +----------------------+
///  y   |                      |
///  |   |                      |
///  |   |                      |
///  V   |                      |
/// nrow |                      |
///      |s|xxxxxxxxxxxxxxxxxx|e|
///      +----------------------+
pub struct CliffWalkingEnv {
    pub ncol: i32,
    pub nrow: i32,
    pub x: i32,
    pub y: i32,
}

impl CliffWalkingEnv {
    pub fn new(ncol: i32, nrow: i32) -> Self {
        assert!(ncol > 0);
        assert!(nrow > 0);
        Self {
            ncol, nrow,
            x: 0, 
            y: nrow as i32 - 1,
        }
    }

    /// 在当前状态下执行 action 动作，更新环境位置，并且返回下一个状态，获得的奖励、是否结束游戏
    pub fn step(&mut self, action: Action) -> ((i32, i32), i32, bool) {
        // 当前状态执行 action 动作
        let delta_pos = action.delta_pos();
        self.x = (self.x + delta_pos.0).max(0).min(self.ncol - 1);
        self.y = (self.y + delta_pos.1).max(0).min(self.nrow - 1);
        let next_state = (self.x, self.y);

        // 奖励，判断是否结束
        let mut reward = -1;
        let mut done = false;
        // 最后一排除了起点，最后一个是重点，其他都是悬崖
        if self.y == self.nrow - 1 && self.x > 0 {
            done = true;
            // 悬崖
            if self.x != self.ncol - 1 {
                reward = -100;
            }
        }

        (next_state, reward, done)
    }

    pub fn reset(&mut self) -> State {
        self.x = 0; 
        self.y = self.nrow as i32 - 1;
        (self.x, self.y)
    }
}

// ==================================================================== //
//                          Sarsa
// ==================================================================== //

pub struct Sarsa {
    /// 在某个状态下，采用某个工作的期望 Q(s, a)
    pub q_table: HashMap<State, Vec<f64>>,
    /// 学习率
    pub alpha: f64,
    /// 折扣因子
    pub gamma: f64,
    /// epsilon-贪婪策略中的参数
    pub epsilon: f64,
}

impl Sarsa {
    pub fn new(ncol: i32, nrow: i32, epsilon: f64, alpha: f64, gamma: f64) -> Self {
        let mut q_table = HashMap::new();
        for x in 0..ncol {
            for y in 0..nrow {
                q_table.insert((x, y),vec![0.0; 4]);
            }
        }    

        Self { q_table, alpha, gamma, epsilon }
    }

    /// 根据 epsilon_policy，根据当前状态，执行一个 action
    pub fn take_action(&self, state: State) -> anyhow::Result<Action> {
        let action: Action = if utils::epsilon_policy(self.epsilon) {
            rand::rng().random_range(0..4).try_into()?
        } else {
            let q_table = self.q_table.get(&state).unwrap();
            let action = utils::arg_max_vec(&q_table) as i32;
            action.try_into()?
        };
        Ok(action)
    }

    /// 在 s0 执行 a0 得到奖励 r，并且进入状态 s1 后又打算执行 a1
    pub fn update(&mut self, s0: State, a0: Action, r: i32, s1: State, a1: Action) {
        /*
            蒙特卡洛：V <- V + alpha * (G - V)
            如果将其中的 V 理解为状态函数：输入某个状态返回这个状态的价值
            
            那么可以将 G 等价为：执行动作的 r +  gamma * 到新状态后的状态价值
            G = r + gamma * V

            蒙特卡洛修改为 TD 方程：
            V(s) <- V(s) + alpha * (r + gamma * V(s+1) - V(s))

            融入 action：

            G(s, a) <- G(s, a) + alpha * (r + gamma * G(s', a') - G(s, a))

            我们要更新工作价值函数：G(s, a)：在状态 s 下，执行工作 a 后回报的期待

            - r：在 (s, a) 又执行后得到的立刻奖励 r
            - G(s', a')：s 采用 a 进入 s' 后，决定执行 a'，这里期望的回报就是 G(s', a')
            - r * gamma * G(s', a')：就是当前奖励 + gamma * 后续的期待奖励
        */

        let g_s1_s1 = *self.q_table.get(&s1).unwrap().get(a1 as usize).unwrap();
        let g_s0_a0 = self.q_table.get_mut(&s0).unwrap().get_mut(a0 as usize).unwrap();
        let td_error = r as f64 + self.gamma * g_s1_s1 - *g_s0_a0;
        *g_s0_a0 += self.alpha * td_error;
    }
}

pub fn test_sarsa() -> anyhow::Result<()> {
    const N_COL: i32 = 12;
    const N_ROW: i32 = 4;
    const N_EPISODES: usize = 500;

    let mut env = CliffWalkingEnv::new(N_COL, N_ROW);
    let mut agent = Sarsa::new(N_COL,N_ROW, 0.1, 0.1, 0.9);

    let mut return_list = vec![];
    for _ in 0..10 {
        for _ in 0..N_EPISODES/10 {
            let mut episode_return = 0;
            // 重新开始游戏
            let mut state = env.reset();
            // 采取一个 action
            let mut action = agent.take_action(state)?;
            let mut done = false;

            while !done {
                // 在环境中执行 action，得到下个状态、奖励和是否结束
                let (next_state, reward, next_done) = env.step(action);
                // 用户现在知道了下一个状态，采样一个动作
                let next_action = agent.take_action(next_state)?;
                episode_return += reward;

                // 用户从 state 采取了 action，并且进入了 next_state，得到 reward，同时用户决定继续采用 next_action
                agent.update(state, action, reward, next_state, next_action);
                
                // 更新
                state = next_state;
                action = next_action;
                done = next_done;
            }

            return_list.push((return_list.len(), episode_return));
        }
    }

    // 绘制奖励
    let root = BitMapBackend::new(&"./result/ch05/sarsa.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(i32::MAX, i32::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(i32::MIN, i32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return (SARSA)", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    Ok(())
}

// ==================================================================== //
//                          NStep Sarsa
// ==================================================================== //

pub struct NStepSarsa {
    /// 在某个状态下，采用某个工作的期望 Q(s, a)
    pub q_table: HashMap<State, Vec<f64>>,
    /// 学习率
    pub alpha: f64,
    /// 折扣因子
    pub gamma: f64,
    /// epsilon-贪婪策略中的参数
    pub epsilon: f64,
    /// 记录数量
    pub n: usize,
    /// 记录
    pub records: VecDeque<(State, Action, i32)>,
}

impl NStepSarsa {
    pub fn new(ncol: i32, nrow: i32, epsilon: f64, alpha: f64, gamma: f64, n: usize) -> Self {
        let mut q_table = HashMap::new();
        for x in 0..ncol {
            for y in 0..nrow {
                q_table.insert((x, y),vec![0.0; 4]);
            }
        }    

        Self { 
            q_table, alpha, gamma, epsilon, n,
            records: VecDeque::new(),
        }
    }

    /// 根据 epsilon_policy，根据当前状态，执行一个 action
    pub fn take_action(&self, state: State) -> anyhow::Result<Action> {
        let action: Action = if utils::epsilon_policy(self.epsilon) {
            rand::rng().random_range(0..4).try_into()?
        } else {
            let q_table = self.q_table.get(&state).unwrap();
            let action = utils::arg_max_vec(&q_table) as i32;
            action.try_into()?
        };
        Ok(action)
    }

    /// 在 s0 执行 a0 得到奖励 r，并且进入状态 s1 后又打算执行 a1
    pub fn update(&mut self, s0: State, a0: Action, r: i32, s1: State, a1: Action, done: bool) {
        /*
            累积几次再更新
        */
        self.records.push_back((s0, a0, r));
        if self.records.len() == self.n {
            // 记录到一定次数，开始更新
            /*
                Saras:
                G(s,a) = G(s,a) + alpha * (r + gamma * G(s',a') - G(s,a))

                连续累积
                G(s,a) = G(s,a) + alpha * (r + g*r' + ... + g*...*g*g*G(s''',a''') - G(s,a))
            */

            // 可以更新 records 中最早的一个！
            let mut ret: f64 = self.q_table[&s1][a1 as usize];        
            for i in (0..self.n).rev() {
                let (s, a, r) = self.records[i]; // 直接通过索引访问
                ret = self.gamma * ret + r as f64;
                
                if done && i > 0 {
                    let q_s_a = self.q_table.get_mut(&s).unwrap().get_mut(a as usize).unwrap();
                    *q_s_a += self.alpha * (ret - *q_s_a);
                }
            }

            let (s, a, _) = self.records.pop_front().unwrap();
            let g_s_a = self.q_table.get_mut(&s).unwrap().get_mut(a as usize).unwrap();
            *g_s_a += self.alpha * (ret - *g_s_a);
        }

        if done {
            self.records.clear();
        }
    }
}

pub fn test_nstep_sarsa() -> anyhow::Result<()> {
    const N_COL: i32 = 12;
    const N_ROW: i32 = 4;
    const N_EPISODES: usize = 500;

    let mut env = CliffWalkingEnv::new(N_COL, N_ROW);
    let mut agent = NStepSarsa::new(N_COL,N_ROW, 0.1, 0.1, 0.9, 5);

    let mut return_list = vec![];
    for _ in 0..10 {
        for _ in 0..N_EPISODES/10 {
            let mut episode_return = 0;
            // 重新开始游戏
            let mut state = env.reset();
            // 采取一个 action
            let mut action = agent.take_action(state)?;
            let mut done = false;

            while !done {
                // 在环境中执行 action，得到下个状态、奖励和是否结束
                let (next_state, reward, next_done) = env.step(action);
                // 用户现在知道了下一个状态，采样一个动作
                let next_action = agent.take_action(next_state)?;
                episode_return += reward;

                // 用户从 state 采取了 action，并且进入了 next_state，得到 reward，同时用户决定继续采用 next_action
                agent.update(state, action, reward, next_state, next_action, next_done);
                
                // 更新
                state = next_state;
                action = next_action;
                done = next_done;
            }

            return_list.push((return_list.len(), episode_return));
        }
    }

    // 绘制奖励
    let root = BitMapBackend::new(&"./result/ch05/nstep_sarsa.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(i32::MAX, i32::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(i32::MIN, i32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return (NStep SARSA)", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    Ok(())
}

// ==================================================================== //
//                          QLearning
// ==================================================================== //

pub struct QLearning {
    /// 在某个状态下，采用某个工作的期望 Q(s, a)
    pub q_table: HashMap<State, Vec<f64>>,
    /// 学习率
    pub alpha: f64,
    /// 折扣因子
    pub gamma: f64,
    /// epsilon-贪婪策略中的参数
    pub epsilon: f64,
}

impl QLearning {
    pub fn new(ncol: i32, nrow: i32, epsilon: f64, alpha: f64, gamma: f64) -> Self {
        let mut q_table = HashMap::new();
        for x in 0..ncol {
            for y in 0..nrow {
                q_table.insert((x, y),vec![0.0; 4]);
            }
        }    

        Self { q_table, alpha, gamma, epsilon }
    }

    /// 根据 epsilon_policy，根据当前状态，执行一个 action
    pub fn take_action(&self, state: State) -> anyhow::Result<Action> {
        let action: Action = if utils::epsilon_policy(self.epsilon) {
            rand::rng().random_range(0..4).try_into()?
        } else {
            let q_table = self.q_table.get(&state).unwrap();
            let action = utils::arg_max_vec(&q_table) as i32;
            action.try_into()?
        };
        Ok(action)
    }

    /// 在 s0 执行 a0 得到奖励 r，并且进入状态 s1
    pub fn update(&mut self, s0: State, a0: Action, r: i32, s1: State) {
        /*
            Sarse 公式：
            G(s, a) <- G(s, a) + alpha * (r + gamma * G(s', a') - G(s, a))
            
            需要进入 s' 后再决定 a'，不太合理，我们直接在 s 状态贪婪选择最好的动作

        */
        let max_next_q = self.q_table[&s1].iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        let g_s0_a0 = self.q_table.get_mut(&s0).unwrap().get_mut(a0 as usize).unwrap();
        let td_error = r as f64 + self.gamma * max_next_q - *g_s0_a0;
        *g_s0_a0 += self.alpha * td_error;
    }
}

pub fn test_qlearning() -> anyhow::Result<()> {
    const N_COL: i32 = 12;
    const N_ROW: i32 = 4;
    const N_EPISODES: usize = 500;

    let mut env = CliffWalkingEnv::new(N_COL, N_ROW);
    let mut agent = QLearning::new(N_COL,N_ROW, 0.1, 0.1, 0.9);

    let mut return_list = vec![];
    for _ in 0..10 {
        for _ in 0..N_EPISODES/10 {
            let mut episode_return = 0;
            // 重新开始游戏
            let mut state = env.reset();
            let mut done = false;

            while !done {
                // 采取一个 action
                let action = agent.take_action(state)?;
                // 在环境中执行 action，得到下个状态、奖励和是否结束
                let (next_state, reward, next_done) = env.step(action);
                episode_return += reward;
                // 用户从 state 采取了 action，并且进入了 next_state，得到 reward
                agent.update(state, action, reward, next_state);
                
                // 更新
                state = next_state;
                done = next_done;
            }

            return_list.push((return_list.len(), episode_return));
        }
    }

    // 绘制奖励
    let root = BitMapBackend::new(&"./result/ch05/qlearning.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let y_min = return_list.iter().map(|(_, y)| *y).fold(i32::MAX, i32::min);
    let y_max = return_list.iter().map(|(_, y)| *y).fold(i32::MIN, i32::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Episode Return (Q-Learning)", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..return_list.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(return_list, &BLUE))?;
    root.present()?;

    Ok(())
}