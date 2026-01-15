use anyhow::Context;
use lumen_core::{FloatDType, Tensor};
use lumen_nn::{functional::nll_loss, init::Init, optim::{Optimizer, SGD}, GCNConv, Module};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    let (x, edge_index, target) = get_karate_club_data().context("get data")?;

    let gnn = KarateGNN::<f32>::new(34, 4, 2, None).context("new model")?;
    let mut optimizer = SGD::new(gnn.params(), 0.1);

    for epoch in 0..1000 {
        let out = gnn.forward(&x, &edge_index).with_context(||format!("{} forward", epoch))?;
        let loss = nll_loss(&out, &target).with_context(||format!("{} loss", epoch))?;
        let grads = loss.backward().with_context(||format!("{} backward", epoch))?;
        optimizer.step(&grads).with_context(||format!("{} step", epoch))?;

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.to_scalar().unwrap());
        }
    }

    Ok(())
}

#[derive(Module)]
pub struct KarateGNN<T: FloatDType> {
    pub conv1: GCNConv<T>,
    pub conv2: GCNConv<T>,

    #[module(skip)]
    pub hidden_dim: usize,
}   

impl<T: FloatDType> KarateGNN<T> {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        init: Option<Init<T>>,
    ) -> anyhow::Result<Self> {
        let conv1 = GCNConv::new(input_dim, hidden_dim, init)?;
        let conv2 = GCNConv::new(hidden_dim, output_dim, init)?;

        Ok(Self {
            conv1,
            conv2,
            hidden_dim,
        })
    }

    pub fn forward(&self, x: &Tensor<T>, edge_index: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
        // (N, input_dim) => (N, hidden_dim)
        let x = self.conv1.forward(x, edge_index).context("conv1 forward")?;
        let x = x.relu(); 
        // (N, hidden_dim) => (N, output_dim)
        let x = self.conv2.forward(&x, edge_index).context("conv2 forward")?;
        // (N, output_dim)
        let x = lumen_nn::functional::log_softmax(&x, 1).context("log softmax")?; 
        Ok(x)
    }
}

pub fn get_karate_club_data() -> anyhow::Result<(Tensor<f32>, Tensor<u32>, Tensor<u32>)> {
    // X
    let num_nodes = 34;
    let x = Tensor::diag(&vec![1.0; num_nodes]).context("create x")?;

    // edge index
    let edges_raw = vec![
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33)
    ];

    let mut srcs = Vec::new();
    let mut dsts = Vec::new();
    for (u, v) in edges_raw {
        // u -> v
        srcs.push(u);
        dsts.push(v);
        // v -> u
        srcs.push(v);
        dsts.push(u);
    }
    assert_eq!(srcs.len(), dsts.len());
    let total_edges = srcs.len(); 
    let mut edge_data = Vec::with_capacity(total_edges * 2);
    edge_data.extend(srcs);
    edge_data.extend(dsts);
    
    let edge_index = Tensor::from_vec(edge_data, (2, total_edges)).context("create edge_index")?;

    // labels
    let labels_vec = vec![
        0, // Node 0
        0, // Node 1
        0, // Node 2
        0, // Node 3
        0, // Node 4
        0, // Node 5
        0, // Node 6
        0, // Node 7
        0, // Node 8
        1, // Node 9 (Officer side)
        0, // Node 10
        0, // Node 11
        0, // Node 12
        0, // Node 13
        1, // Node 14
        1, // Node 15
        0, // Node 16
        0, // Node 17
        1, // Node 18
        0, // Node 19
        1, // Node 20
        0, // Node 21
        1, // Node 22
        1, // Node 23
        1, // Node 24
        1, // Node 25
        1, // Node 26
        1, // Node 27
        1, // Node 28
        1, // Node 29
        1, // Node 30
        1, // Node 31
        1, // Node 32
        1  // Node 33 (The Officer)
    ];
    let labels = Tensor::from_vec(labels_vec, (num_nodes, 1)).context("create labels")?;

    Ok((x, edge_index, labels))
}