use lumen_core::*;
use lumen_nn::{criterion::MSELoss, model::MLP, optim::Momentum, Criterion, Model, Optim};

fn main() {
    let input = Tensor::build([
        0., 0.,
        0., 1.,
        1., 0.,
        1., 1.,
    ], [4, 2]).unwrap().require_grad();

    let target = Tensor::build([
        0.,
        1.,
        1.,
        0.,
    ], [4, 1]).unwrap().require_grad();

    let mlp = MLP::from_archs([2, 4, 1]).unwrap();
    let mut optimizer = Momentum::new(mlp.parameters(), 0.1, 0.1);
    let criterion = MSELoss::new();

    for _ in 0..5000 {
        optimizer.zero_grad();
        let pred = mlp.forward(&input).unwrap();
        let loss = criterion.loss(&pred, &target).unwrap();
        loss.backward();
        optimizer.step();
    }
}