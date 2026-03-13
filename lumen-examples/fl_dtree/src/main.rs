mod client;
mod server;
pub use client::*;
pub use server::*;
use anyhow::Context;
use lumen_core::Tensor;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {e:?}");
    }
}

fn result_main() -> anyhow::Result<()> {
    const N_CLIENTS: usize = 3;
    const N_SAMPLES: usize = 300;
    const N_FEATURES: usize = 2;
    const TEST_RATIO: f32 = 0.2;
    const MAX_DEPTH: usize = 3;

    let ((train_x, train_y), (test_x, test_y)) = generate_client_data(N_SAMPLES, N_FEATURES, TEST_RATIO).context("generate_client_data")?;
    
    let server = FederatedServer::new(train_x, train_y, N_CLIENTS).context("init server")?;
    let tree = server.build_tree(MAX_DEPTH).context("build tree")?;

    let predict = tree.predict(&test_x)?;
    let real = test_y.to_vec()?;

    let total = predict.len();
    let right_count = predict.into_iter().zip(real).filter(|(p, r)| p == r).count();    

    println!("🚀 联邦单 CART 树在全局测试集上的准确率: {}", right_count as f32 / total as f32);
    Ok(())
}

fn generate_client_data(n_samples: usize, n_features: usize, test_ratio: f32) -> anyhow::Result<((Tensor<f32>, Tensor<bool>), (Tensor<f32>, Tensor<bool>))> {
    assert!(test_ratio > 0.0);
    assert!(test_ratio < 1.0);

    // 随即输出 x
    let x = Tensor::<f32>::randn(0.0, 1.0, (n_samples, n_features))?;
    // 计算 y: 所有 x 求和 > 0 为 true，否则为 false
    let sum_x = x.sum(1)?; // (n_samples, )
    let y = sum_x.ge(0.0)?; // (n_samples, )

    // 切分
    let train_size = (n_samples as f32 * test_ratio) as usize;
    let test_size = n_samples - train_size;
    let train_x = x.narrow(0, 0, train_size)?;
    let test_x = x.narrow(0, train_size, test_size)?;
    let train_y = y.narrow(0, 0, train_size)?;
    let test_y = y.narrow(0, train_size, test_size)?;
    
    Ok(((train_x, train_y), (test_x, test_y)))
}