# Lumen

Tensor library for machine learning like [PyTorch](https://pytorch.org/).


## Usages

Use `Tensor` like PyTroch：

````rust
use lumen_core::*;
let t = Tensor::from([[1, 2], [3, 4]]);
let zt = t.zeros_like();
let t = Tensor::rand([5, 4]);

let t = Tensor::build(vec![1., 2., 3., 4.], [2, 2]).unwrap();
let col_t = t.reshape([1, 4]).unwrap();
let row_t = t.reshape([4, 1]).unwrap();
````

Build a network use `lumen_nn` just like PyTroch：

````rust
use lumen_core::*;
use lumen_dataset::{IrisDataSet, DataLoader};
use lumen_nn::{Criterion, Model, Optim};
use lumen_nn::criterion::MSELoss;
use lumen_nn::optim::SDG;
use lumen_nn::model::MLP;

let dataset = IrisDataSet::new();
let dataloader = DataLoader::new(dataset, 16, true);

let model = MLP::from_archs(vec![4, 10, 3]).unwrap();
let criterion = MSELoss::new();
let mut optimizer = SDG::new(model.parameters(), 0.01);

const EPOCHS: usize = 100;
for epoch in 0..EPOCHS {
    for (i, (x, y)) in dataloader.iter().enumerate() {
        optimizer.zero_grad();
        let pred = model.forward(&x).unwrap();
        let loss = criterion.loss(&pred, &y).unwrap();
        loss.backward();
        optimizer.step();

        if i == 0 && epoch % 10 == 0 {
            println!("Epoch: {epoch}/{EPOCHS}, Loss: {:?}", loss.mean());
        }
    }
}
````