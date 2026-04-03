use lumen_core::{IndexOp, Tensor};
use serde::{Deserialize, Serialize};

pub const N_FEATURES: usize = 2;

#[derive(Serialize, Deserialize, Debug)]
pub enum ServerMsg {
    /// 初始化根节点，让客户端生成全是 True 的 mask 存入 node_id
    InitRoot { node_id: usize },
    /// 询问某个节点的 0/1 标签分布
    GetCounts { node_id: usize },
    /// 询问在某个划分下的左右分布
    EvaluateSplit { node_id: usize, feature_index: usize, threshold: f32 },
    /// 确认划分，让客户端生成左右子节点的 mask 并保存
    ApplySplit { 
        node_id: usize, 
        feature_index: usize, 
        threshold: f32, 
        left_node_id: usize, 
        right_node_id: usize 
    },
    /// 训练结束
    Stop,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ClientMsg {
    Ack,
    Counts(usize, usize), // (zero_count, false_count/one_count)
    SplitEvaluation {
        left_counts: (usize, usize),
        right_counts: (usize, usize),
    },
}

pub fn generate_data(n_samples: usize, n_features: usize) -> anyhow::Result<(Tensor<f32>, Tensor<bool>)> {
    // 随即输出 x
    let x = Tensor::<f32>::rand(-1.1, 1.1, (n_samples, n_features))?;
    // 计算 y: 所有 x 求和 > 0 为 true，否则为 false
    let sum_x = x.sum(1)?; // (n_samples, )
    let y = sum_x.ge(0.0)?; // (n_samples, )

    Ok((x, y))
}

pub enum DecisionTree {
    Root(bool),
    Node {
        feature_index: usize,
        threshold: f32,
        left: Box<DecisionTree>,
        right: Box<DecisionTree>,
    }
}

impl DecisionTree {
    pub fn root(value: bool) -> Self {
        Self::Root(value)
    }

    pub fn node(
        feature_index: usize,
        threshold: f32,
        left: Box<DecisionTree>,
        right: Box<DecisionTree>,
    ) -> Self {
        Self::Node { feature_index, threshold, left, right }
    }

    // (n_samples, n_features)
    pub fn predict(&self, xs: &Tensor<f32>) -> anyhow::Result<Vec<bool>> {
        let (n_samples, _) = xs.dims2()?;
        let mut results = vec![];
        for sample in 0..n_samples {
            let x = xs.index(sample)?;
            let result = self.predict_one(&x)?;
            results.push(result);
        }
        Ok(results)
    }

    // x: (n_features)
    pub fn predict_one(&self, x: &Tensor<f32>) -> anyhow::Result<bool> {
        match self {
            Self::Root(v) => Ok(*v),
            Self::Node { feature_index, threshold, left, right } => {
                if x.index(*feature_index)?.to_scalar()? <= *threshold {
                    left.predict_one(x)
                } else {
                    right.predict_one(x)
                }
            }
        }
    }
}