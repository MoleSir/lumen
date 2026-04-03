use lumen_core::{IndexOp, Tensor};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use fl_dtree_tcp::{generate_data, ClientMsg, ServerMsg, N_FEATURES};

pub struct Client {
    pub id: usize,
    pub x_local: Tensor<f32>,
    pub y_local: Tensor<bool>,
    pub masks: HashMap<usize, Tensor<bool>>,
}

impl Client {
    pub fn new(id: usize, x_local: Tensor<f32>, y_local: Tensor<bool>) -> Self {
        Self { id, x_local, y_local, masks: HashMap::new() }
    }

    pub fn run_tcp_worker(&mut self, addr: &str) -> anyhow::Result<()> {
        let stream = TcpStream::connect(addr)?;
        println!("Client {} connected to server", self.id);
        
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut writer = stream;

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 { 
                println!("[Client {}] 服务端已断开连接，客户端安全退出。", self.id);
                break; 
            }

            let msg: ServerMsg = serde_json::from_str(&line)?;
            let reply = self.handle_msg(msg)?;
            
            let reply_str = serde_json::to_string(&reply)? + "\n";
            writer.write_all(reply_str.as_bytes())?;
        }
        Ok(())
    }

    fn handle_msg(&mut self, msg: ServerMsg) -> anyhow::Result<ClientMsg> {
        match msg {
            ServerMsg::InitRoot { node_id } => {
                let shape = self.x_local.dim(0)?;
                let mask = Tensor::trues(shape)?;
                self.masks.insert(node_id, mask);
                Ok(ClientMsg::Ack)
            }
            ServerMsg::GetCounts { node_id } => {
                let mask = &self.masks[&node_id];
                let valid_y = self.y_local.index(mask)?;
                Ok(ClientMsg::Counts(valid_y.true_count()?, valid_y.false_count()?))
            }
            ServerMsg::EvaluateSplit { node_id, feature_index, threshold } => {
                let mask = &self.masks[&node_id];
                let valid_x = self.x_local.index(mask)?;
                let valid_y = self.y_local.index(mask)?;
                
                let left_condition = valid_x.index((.., feature_index))?.ge(threshold)?;
                let right_condition = left_condition.not()?;

                let left_y = valid_y.index(left_condition)?;
                let right_y = valid_y.index(right_condition)?;

                Ok(ClientMsg::SplitEvaluation {
                    left_counts: (left_y.true_count()?, left_y.false_count()?),
                    right_counts: (right_y.true_count()?, right_y.false_count()?),
                })
            }
            ServerMsg::ApplySplit { node_id, feature_index, threshold, left_node_id, right_node_id } => {
                let mask = &self.masks[&node_id];
                let condition = self.x_local.index((.., feature_index))?.ge(threshold)?;
                
                let left_mask = mask.and(&condition)?;
                let right_mask = mask.and(&condition.not()?)?;
                
                // 存入新的子节点 mask，并可以释放当前父节点 mask 节省内存
                self.masks.insert(left_node_id, left_mask);
                self.masks.insert(right_node_id, right_mask);
                self.masks.remove(&node_id);
                
                Ok(ClientMsg::Ack)
            }
            ServerMsg::Stop => {
                println!("[Client {}] 收到停止指令，准备退出...", self.id);
                Ok(ClientMsg::Ack)
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    const N_SAMPLES: usize = 300;

    let args: Vec<String> = std::env::args().collect();
    let client_id: usize = args.get(1).unwrap_or(&"0".to_string()).parse()?;
    
    let (x, y) = generate_data(N_SAMPLES, N_FEATURES)?;
    let mut client = Client::new(client_id, x, y);
    
    client.run_tcp_worker("127.0.0.1:8888")?;
    Ok(())
}