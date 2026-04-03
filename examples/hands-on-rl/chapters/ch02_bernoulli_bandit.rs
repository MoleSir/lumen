use std::path::Path;
use anyhow::Context;
use lumen_core::{IndexOp, Tensor};
use plotters::{chart::ChartBuilder, prelude::{BitMapBackend, IntoDrawingArea}, series::LineSeries, style::{BLUE, WHITE}};
use rand::RngExt;
use hands_on_rl::utils;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    let bandit_10_arm = BernoulliBandit::new(10)?;
    let policy = EpsilonGreedy::new(&bandit_10_arm, 0.01)?;
    run_solve_and_plot(bandit_10_arm.clone(), policy, 5000, "./result/ch02/epsilon_greedy.png")?;

    let policy = DecayingEpsilonGreedy::new(&bandit_10_arm)?;
    run_solve_and_plot(bandit_10_arm.clone(), policy, 5000, "./result/ch02/decaying_epsilon_greedy.png")?;

    let policy = UCB::new(&bandit_10_arm, 1.0)?;
    run_solve_and_plot(bandit_10_arm.clone(), policy, 5000, "./result/ch02/ucb.png")?;

    Ok(())
}

// ================================================================================= //
//              BernoulliBandit
// ================================================================================= //

#[derive(Clone)]
pub struct BernoulliBandit {
    pub k: usize,
    pub probs: Tensor<f64>, // (k, )
    pub best_idx: usize,
    pub best_prob: f64,
}

impl BernoulliBandit {
    pub fn new(k: usize) -> anyhow::Result<Self> {
        let probs = Tensor::rand(0.0, 1.0, (k, ))?; // (k,)
        let best_idx = probs.argmax(0)?.to_scalar()? as usize; // (,)
        let best_prob = probs.index(best_idx)?.to_scalar()?;
        Ok(Self {
            probs, best_idx, best_prob, k
        })
    }

    pub fn prob(&self, i: usize) -> anyhow::Result<f64> {
        let v = self.probs.index(i)?.to_scalar()?;
        Ok(v)
    }

    pub fn step(&self, i: usize) -> anyhow::Result<u32> {
        let prob = self.prob(i)?;
        let value = rand::rng().random_range(0.0..1.0);
        Ok(if value < prob {1} else {0})
    }  
}

// ================================================================================= //
//              Solver
// ================================================================================= //

pub struct SolverState {
    pub bandit: BernoulliBandit,
    /// 每根拉杆的尝试次数
    pub counts: Vec<usize>,
    /// 当前步的累积懊悔
    pub regret: f64,
    /// 维护一个列表,记录每一步的动作
    pub actions: Vec<usize>,
    /// 维护一个列表,记录每一步的累积懊悔
    pub regrets: Vec<f64>,
}

pub struct Solver<P> {
    /// solver 状态
    pub state: SolverState,
    /// solver 策略
    pub policy: P,
}

pub trait SolvePolicy : Sized {
    fn run_one_step(&mut self, state: &SolverState) -> anyhow::Result<usize>;
}

impl<P: SolvePolicy> Solver<P> {
    pub fn new(bandit: BernoulliBandit, policy: P) -> anyhow::Result<Self> {
        let counts = vec![0; bandit.k];
        Ok(Self { 
            state: SolverState {
                bandit,
                counts,
                regret: 0.0,
                actions: vec![],
                regrets: vec![],
            },
            policy,
        })
    }

    pub fn update_regret(&mut self, i: usize) -> anyhow::Result<()> {
        self.state.regret += self.state.bandit.best_prob - self.state.bandit.prob(i)?;
        self.state.regrets.push(self.state.regret);
        Ok(())
    }

    pub fn run(&mut self, num_steps: usize) -> anyhow::Result<()> {
        for _ in 0..num_steps {
            let i = self.policy.run_one_step(&self.state)?;
            assert!(i < self.state.bandit.k);
            self.state.counts[i] += 1;
            self.state.actions.push(i);
            self.update_regret(i)?;
        }
        Ok(())
    }
}

pub fn run_solve_and_plot<P: SolvePolicy>(
    bandit: BernoulliBandit, 
    policy: P, 
    num_steps: usize,
    output_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let mut solver = Solver::new(bandit, policy).context("new solver")?;
    solver.run(num_steps).context("run steps")?;

    let root = BitMapBackend::new(&output_path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let data: Vec<(usize, f64)> = solver.state.regrets.iter().cloned().enumerate().collect();

    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Regret Curve", ("sans-serif", 20))
        .build_cartesian_2d(0..data.len(), y_min..y_max)?;
    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(data, &BLUE))?;
    root.present()?;
    Ok(())
}

// ================================================================================= //
//              EpsilonGreedy
// ================================================================================= //

pub struct EpsilonGreedy {
    epsilon: f64,
    estimates: Vec<f64>, // (k)
}

impl EpsilonGreedy {
    pub fn new(bandit: &BernoulliBandit, epsilon: f64) -> anyhow::Result<Self> {
        let estimates = vec![1.0; bandit.k];
        Ok(Self { epsilon, estimates })   
    }
}

impl SolvePolicy for EpsilonGreedy {
    fn run_one_step(&mut self, state: &SolverState) -> anyhow::Result<usize> {
        // 随机一个值来决定是探索还是用之前最好的结果！
        let rvalue = rand::rng().random_range(0.0..1.0);
        let i = if rvalue < self.epsilon {
            rand::rng().random_range(0..state.bandit.k)
        } else {
            utils::arg_max_vec(&self.estimates)
        };

        let r = state.bandit.step(i)?;
        // 根据接受的奖励更新评估值
        self.estimates[i] += 1. / (state.counts[i] as f64 + 1.) * (r as f64 - self.estimates[i]);

        Ok(i)
    }
}

// ================================================================================= //
//              DecayingEpsilonGreedy
// ================================================================================= //

pub struct DecayingEpsilonGreedy {
    estimates: Vec<f64>, // (k)
    count: usize,
}

impl DecayingEpsilonGreedy {
    pub fn new(bandit: &BernoulliBandit) -> anyhow::Result<Self> {
        let estimates = vec![1.0; bandit.k];
        Ok(Self { count: 0, estimates })   
    }
}

impl SolvePolicy for DecayingEpsilonGreedy {
    fn run_one_step(&mut self, state: &SolverState) -> anyhow::Result<usize> {
        self.count += 1;

        // 随机一个值来决定是探索还是用之前最好的结果！
        // epsilon = 1/self.cout: epsilon 越来越小，表示探索的概率从一开始很大到很小！
        let rvalue = rand::rng().random_range(0.0..1.0);
        let i = if rvalue < 1.0 / self.count as f64 {
            rand::rng().random_range(0..state.bandit.k)
        } else {
            utils::arg_max_vec(&self.estimates)
        };

        let r = state.bandit.step(i)?;
        // 根据接受的奖励更新评估值
        self.estimates[i] += 1. / (state.counts[i] as f64 + 1.) * (r as f64 - self.estimates[i]);

        Ok(i)
    }
}


// ================================================================================= //
//              UCB
// ================================================================================= //

pub struct UCB {
    estimates: Vec<f64>, // (k)
    count: usize,
    coef: f64,
}

impl UCB {
    pub fn new(bandit: &BernoulliBandit, coef: f64) -> anyhow::Result<Self> {
        let estimates = vec![1.0; bandit.k];
        Ok(Self { count: 0, estimates, coef })   
    }
}

impl SolvePolicy for UCB {
    fn run_one_step(&mut self, state: &SolverState) -> anyhow::Result<usize> {
        self.count += 1;

        /*
        
        np.sqrt(
            np.log(self.total_count) 
            / 
            (2 * (self.counts + 1))
        )

        */
        // self.coef * ((self.count as f64).ln() / (2.0 * (state.counts)));

        let ucb: Vec<f64> = self.estimates.iter().zip(state.counts.iter())
            .map(|(&estimates, &count)| {
                estimates + self.coef * ((self.count as f64).ln() / (2.0 * (count as f64 + 1.0))).sqrt()
            })
            .collect();
        let i = utils::arg_max_vec(&ucb);
        let r = state.bandit.step(i)?;
        // 根据接受的奖励更新评估值
        self.estimates[i] += 1. / (state.counts[i] as f64 + 1.) * (r as f64 - self.estimates[i]);

        Ok(i)
    }
}

