use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use anyhow::Context;
use fl_dtree_tcp::{generate_data, ClientMsg, DecisionTree, ServerMsg, N_FEATURES};

pub struct Server {
    pub n_features: usize,
    pub clients: Vec<TcpStream>,
    pub next_node_id: usize,
}

impl Server {
    pub fn new(n_features: usize, n_clients: usize, port: u16) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port))?;
        println!("Server listening on port {}, waiting for {} clients...", port, n_clients);
        
        let mut clients = vec![];
        for stream in listener.incoming().take(n_clients) {
            clients.push(stream?);
            println!("Client connected!");
        }

        Ok(Self { n_features, clients, next_node_id: 0 })
    }

    /// 辅助函数：发送消息给所有客户端并收集回复
    fn broadcast_msg(&mut self, msg: &ServerMsg) -> anyhow::Result<Vec<ClientMsg>> {
        let msg_str = serde_json::to_string(msg)? + "\n";
        let mut replies = vec![];
        
        for stream in self.clients.iter_mut() {
            stream.write_all(msg_str.as_bytes())?;
            
            let mut reader = BufReader::new(stream.try_clone()?);
            let mut line = String::new();
            reader.read_line(&mut line)?;
            
            let reply: ClientMsg = serde_json::from_str(&line)?;
            replies.push(reply);
        }
        Ok(replies)
    }

    pub fn build_tree(&mut self, max_depth: usize) -> anyhow::Result<Box<DecisionTree>> {
        let root_id = self.next_node_id;
        self.next_node_id += 1;

        // 初始化客户端的根节点 mask
        self.broadcast_msg(&ServerMsg::InitRoot { node_id: root_id })?;

        let tree = self.do_build_tree(max_depth, 0, root_id)?;
        
        self.broadcast_msg(&ServerMsg::Stop)?;

        Ok(tree)
    }

    fn do_build_tree(&mut self, max_depth: usize, cur_depth: usize, node_id: usize) -> anyhow::Result<Box<DecisionTree>> {
        // 1. 获取当前节点标签分布
        let replies = self.broadcast_msg(&ServerMsg::GetCounts { node_id })?;
        let mut global_counts = (0, 0);
        for reply in replies {
            if let ClientMsg::Counts(c0, c1) = reply {
                global_counts.0 += c0;
                global_counts.1 += c1;
            }
        }

        let total_samples = global_counts.0 + global_counts.1;
        println!("  [深度 {}] 节点 ID {} 总样本: {}, 分布: {:?}", cur_depth, node_id, total_samples, global_counts);

        if cur_depth >= max_depth || global_counts.0 == 0 || global_counts.1 == 0 {
            let pred_class = if global_counts.0 > global_counts.1 { false } else { true };
            println!("    -> 生成叶子节点，预测类别: {pred_class}");
            return Ok(Box::new(DecisionTree::root(pred_class)));
        }

        // 2. 寻找最优划分
        let mut best_gini = f32::MAX;
        let mut best_criteria = None;

        for feature_index in 0..self.n_features {
            for threshold in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                let replies = self.broadcast_msg(&ServerMsg::EvaluateSplit { node_id, feature_index, threshold })?;
                
                let mut global_left = (0, 0);
                let mut global_right = (0, 0);
                
                for reply in replies {
                    if let ClientMsg::SplitEvaluation { left_counts, right_counts } = reply {
                        global_left.0 += left_counts.0; global_left.1 += left_counts.1;
                        global_right.0 += right_counts.0; global_right.1 += right_counts.1;
                    }
                }

                let gini = Self::calc_split_gini(global_left, global_right);
                if gini < best_gini {
                    best_gini = gini;
                    best_criteria = Some((feature_index, threshold));
                }
            }
        }

        // 3. 执行划分
        let (best_feat, best_thresh) = best_criteria.unwrap();
        println!("    -> 最优特征: X{}, 阈值: {}, 基尼: {}", best_feat, best_thresh, best_gini);

        let left_node_id = self.next_node_id;
        let right_node_id = self.next_node_id + 1;
        self.next_node_id += 2;

        self.broadcast_msg(&ServerMsg::ApplySplit { 
            node_id, feature_index: best_feat, threshold: best_thresh, left_node_id, right_node_id 
        })?;

        // 递归
        let left_child = self.do_build_tree(max_depth, cur_depth+1, left_node_id)
            .with_context(|| format!("build left tree in depth {}", cur_depth+1))?;
        let right_child = self.do_build_tree(max_depth, cur_depth+1, right_node_id)
            .with_context(|| format!("build righttree in depth {}", cur_depth+1))?;

        Ok(Box::new(DecisionTree::node(best_feat, best_thresh, left_child, right_child)))
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

fn main() -> anyhow::Result<()> {
    let mut server = Server::new(N_FEATURES, 3, 8888)?;
    let tree = server.build_tree(3)?;

    // 测试
    let (test_x, test_y) = generate_data(50, N_FEATURES)?;
    let predict = tree.predict(&test_x)?;
    let real = test_y.to_vec()?;

    let total = predict.len();
    let right_count = predict.into_iter().zip(real).filter(|(p, r)| p == r).count();    

    println!("🚀 联邦单 CART 树在全局测试集上的准确率: {}", right_count as f32 / total as f32);
    
    Ok(())
}