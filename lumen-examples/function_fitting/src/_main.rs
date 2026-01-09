use lumen_core::*;
use lumen_dataset::{DataSet, DataLoader};
use lumen_nn::criterion::MSELoss;
use lumen_nn::model::{Linear, RNN};
use lumen_nn::optim::{Adam, AdamConfig};
use lumen_nn::{Criterion, Model, Optim};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;
use anyhow::Result;

struct FunctionDataSet {
    func: fn(f64) -> f64,
    num_samples: usize,
}

impl FunctionDataSet {
    pub fn new(func: fn(f64) -> f64, num_samples: usize) -> Self {
        Self { func, num_samples }
    }
}

impl DataSet for FunctionDataSet {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, _: usize) -> (Tensor, Tensor) {
        const MIN_FUNC_X: f64 = 0.;
        const MAX_FUNC_X: f64 = 2. * std::f64::consts::PI;
        
        let seq_len = 100;
        let x = Tensor::linspace(MIN_FUNC_X, MAX_FUNC_X, seq_len);
        let y = x.map(self.func);

        (x.view([seq_len, 1]).unwrap(), y.view([seq_len, 1]).unwrap())
    }
}

#[allow(unused)]
struct FunctionModel {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    rnn: RNN,
    fc1: Linear,
    fc2: Linear,
}

impl FunctionModel {
    pub fn new(hidden_size: usize) -> Self {
        let input_size = 1;
        let output_size = 1;
        Self {
            input_size,
            hidden_size,
            output_size,
            rnn: RNN::new(input_size, hidden_size),
            fc1: Linear::new(hidden_size, 2*hidden_size, true),
            fc2: Linear::new(2*hidden_size, output_size, true)
        }
    }    

    fn step(&self, r_out: &Tensor) -> Result<Tensor> {
        // r_out: each seq after rnn (batch_size, hidden_size)
        let out = op::tanh(&self.fc1.forward(r_out)?);
        self.fc2.forward(&out)
    }
}

impl Model for FunctionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input: (batch_size, seq_len, hidden_size)
        let seq_len = input.shape()[1];

        let r_outs = self.rnn.forward(&input, None)?;
        assert!(r_outs.requires_grad());

        let outs: Vec<_> = (0..seq_len).map(|time| {
            let view = r_outs.slice(rngs!(time)).unwrap().require_grad();
            self.step(&view).unwrap()
        }).collect();
        
        // output: (seq_len, batch_size, output_size)
        op::stack(&outs)
    }

    fn parameters(&self) -> impl Iterator<Item = &Tensor> {
        self.rnn.parameters().chain(self.fc1.parameters()).chain(self.fc2.parameters())
    }
}

type FunctionDataLoader = DataLoader<FunctionDataSet>;

fn get_dataloader(func: fn(f64)->f64, train_samples: usize, test_sample: usize) 
    -> (FunctionDataLoader, FunctionDataLoader)
{
    let train_dataset = FunctionDataSet::new(func, train_samples);
    let train_dataloader = FunctionDataLoader::new(train_dataset, 1, true);

    let test_dataset = FunctionDataSet::new(func, test_sample);
    let test_dataloader = FunctionDataLoader::new(test_dataset, 1, true);

    (train_dataloader, test_dataloader)
}

fn main_result() -> Result<()> {
    println!("Function fitting~");
    const TRAIN_SMAPLES: usize = 20000;
    const TEST_SAMPLES: usize = 1;

    fn func(x: f64) -> f64 {
        x.cos()
    }

    let (train_dataloader, test_dataloader) = get_dataloader(func, TRAIN_SMAPLES, TEST_SAMPLES);
    let model = FunctionModel::new(5);
    let cirterion = MSELoss::new();
    let mut optimizer = Adam::with_config(model.parameters(), AdamConfig::default());

    for (i, (x, y)) in train_dataloader.iter().enumerate() {
        let seq_len = x.shape()[1];
        let y = y.view([seq_len, 1, 1]).unwrap();
        let pred = model.forward(&x).unwrap();
        let loss = cirterion.loss(&pred, &y).unwrap();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if i % 100 == 0 {
            print!("Loss: {:?}\n", loss.mean());
        }
    }

    let (x, _) = test_dataloader.iter().next().unwrap();
    let pred = model.forward(&x).unwrap();
    let data = x.iter().zip(pred.iter()).map(|(&a, &b)| (a, b)).collect();
    let plot = Plot::new(data).line_style(LineStyle::new().colour("red"));
    let view = ContinuousView::new().add(plot);
    Page::single(&view).save("image/plot.svg").unwrap();

    Ok(())
}

fn main() {
    if let Err(err) = main_result() {
        eprintln!("{:?}", err);
    }
}