use anyhow::Context;
use lumen_core::{IndexOp, Tensor};
use crate::Client;

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

pub struct FederatedServer {
    pub n_features: usize,
    pub candidate_thresholds: Vec<Vec<f32>>,
    pub clients: Vec<Client>,
}

impl FederatedServer {
    /// x: (n_samples, n_features)
    /// y: (n_samples,)
    pub fn new(x: Tensor<f32>, y: Tensor<bool>, n_client: usize) -> anyhow::Result<Self> {
        // split
        let (n_samples, n_features) = x.dims2()?;
        if n_samples % n_client != 0 {
            return Err(anyhow::anyhow!("n_samples({}) % client_count({}) != 0", n_samples, n_client));
        }
        let n_client_samples = n_samples / n_client;

        // (n_samples / client_count, n_features)
        let mut clients = vec![];
        for i in 0..n_client {
            let x_local = x.narrow(0, i*n_client_samples, n_client_samples)?;
            let y_local = y.narrow(0, i*n_client_samples, n_client_samples)?;
            clients.push(Client::new(i, x_local, y_local));
        }

        // 简化模拟，假设服务器事先知道特征的搜索范围 (比如分 10 个档位)
        let candidate_thresholds = (0..n_features)
            .map(|_| {
                Tensor::<f32>::linspace(-3.0, 3.0, 10).expect("linspace").to_vec().unwrap()
            })
            .collect();
        
        Ok(Self { n_features, candidate_thresholds, clients })
    }

    pub fn build_tree(&self, max_depth: usize) -> anyhow::Result<Box<DecisionTree>> {
        let mut initial_masks = vec![];
        for (i, client) in self.clients.iter().enumerate() {
            let shape = client.x_local.dim(0)?;
            let mask = Tensor::trues(shape).with_context(|| format!("init client {}'s mask", i))?;
            initial_masks.push(mask);
        }

        self.do_build_tree(max_depth, 0, initial_masks).with_context(|| format!("build_tree in depth 0"))
    }

    fn do_build_tree(&self, max_depth: usize, cur_depth: usize, client_masls: Vec<Tensor<bool>>) -> anyhow::Result<Box<DecisionTree>> {
        assert_eq!(self.clients.len(), client_masls.len());

        // 向客户端询问当前节点的全局标签分布
        // 对每个 client，在 mask[i] 的条件下，计算这些样本输出 0 / 1 的数量，并将三个客户端 0/1 数量求和
        let mut global_counts = (0, 0);

        for (i, (client, mask)) in self.clients.iter().zip(client_masls.iter()).enumerate() {
            let client_counts = client
                .get_counts(mask)
                .with_context(|| format!("client {i} get counts"))?;

            global_counts.0 += client_counts.0;
            global_counts.1 += client_counts.1;
        }

        let total_samples = global_counts.0 + global_counts.1;
        println!("  [深度 {}] 当前节点总样本数: {}, 分布: {:?}", cur_depth, total_samples, global_counts);
        
        // 停止条件：达到最大深度，或者节点纯了（只有一类，说明按照当前的划分方式全部都分到一边去了）
        if cur_depth >= max_depth || global_counts.0 == 0 || global_counts.1 == 0 {
            let pred_class = if global_counts.0 > global_counts.1 { false } else { true };
            println!("    -> 生成叶子节点，预测类别: {pred_class}");
            return Ok(Box::new(DecisionTree::root(pred_class)));
        }
        
        let mut best_gini = f32::MAX;
        let mut best_criteria = None;

        assert_eq!(self.candidate_thresholds.len(), self.n_features);
        for feature_index in 0..self.n_features {
            for &threshold in self.candidate_thresholds[feature_index].iter() {
                // 对 feature_index 这个特征进行 thresh 划分 
                let mut global_left_counts = (0, 0);
                let mut global_right_counts = (0, 0);

                // 通信环节：向所有客户端询问基于 (feature_index, thresh) 的切分分布
                for (i, (client, mask)) in self.clients.iter().zip(client_masls.iter()).enumerate() {
                    /*
                    对 i 个 client，让他计算下自己数[据中：满足在 client_masks[i] 的样本中，
                    - 满足在 f_idx 特征 <= thresh 的样本的 0 / 1 输出数量
                    - 满足在 f_idx 特征 >  thresh 的样本的 0 / 1 输出数量
                    */
                    let [l_counts, r_counts]  = client
                        .evaluate_split(mask, feature_index, threshold)
                        .with_context(|| format!("client {i} evaluate_split"))?;

                    global_left_counts.0 += l_counts.0;
                    global_left_counts.1 += l_counts.1;   

                    global_right_counts.0 += r_counts.0;
                    global_right_counts.1 += r_counts.1;   
                }

                // global_left_counts: 所有样本满足在 f_idx 特征 <= thresh 的样本的 0 / 1 输出数量
                
                // 服务器计算这个切分的全局基尼系数
                let gini = Self::calc_split_gini(global_left_counts, global_right_counts);

                // 更新 best
                if gini < best_gini {
                    best_gini = gini;
                    best_criteria = Some((feature_index, threshold));
                }
            }
        }

        // 确定最优切分点后，通知客户端更新归属，继续递归建树
        let (best_feature_index, best_threshold) = best_criteria.expect("no best!???");
        println!("    -> 决定分裂！最优特征: X{}, 阈值: {:2}, 基尼系数: {:4}", best_feature_index, best_threshold, best_gini);

        let mut left_mask = vec![];
        let mut right_mask = vec![];
        for (i, (client, mask)) in self.clients.iter().zip(client_masls.iter()).enumerate() {
            /*
                在原来 client_masks[i] 过滤的基础上，左侧增加满足 <= best_thresh 的，右侧增加 > best_thresh 样本
                就是说，经过 client_masks[i] 过滤的剩余样本，将满足 <= best_thresh 分到 l_mask，其他分到 r_mask
            */
            let (l_mask, r_mask) = client
                .apply_split(mask, best_feature_index, best_threshold)
                .with_context(|| format!("client {i} apply_split"))?;

            left_mask.push(l_mask);
            right_mask.push(r_mask);
        }

        let left_child = self.do_build_tree(max_depth, cur_depth+1, left_mask).with_context(|| format!("build left tree in depth {}", cur_depth+1))?;
        let right_child = self.do_build_tree(max_depth, cur_depth+1, right_mask).with_context(|| format!("build righttree in depth {}", cur_depth+1))?;

        Ok(Box::new(DecisionTree::node(best_feature_index, best_threshold, left_child, right_child)))
    }

    fn calc_gini(counts: (usize, usize)) -> f32 {
        let total = counts.0 + counts.1;
        if total == 0 {
            return 0.0;
        }
        let prob0 = counts.0 as f32 / total as f32;
        let prob1 = counts.1 as f32 / total as f32;
        
        1.0 - (prob0.powi(2) + prob1.powi(2))
    } 

    fn calc_split_gini(left_counts: (usize, usize), right_counts: (usize, usize)) -> f32 {
        let total_left = left_counts.0 + left_counts.1;
        let total_right = right_counts.0 + right_counts.1;
        let total = total_left + total_right;
        
        if total == 0 {
            return 0.0;
        } 

        let gini_left = Self::calc_gini(left_counts);
        let gini_right = Self::calc_gini(right_counts);
    
        let scale_left = total_left as f32 / total as f32;
        let scale_right = total_right as f32 / total as f32;

        scale_left * gini_left + scale_right * gini_right
    }
}