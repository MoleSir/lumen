use lumen_core::{FloatDType, IndexOp, Tensor};
use lumen_dataset::{Batcher, DataLoader, Dataset};
use lumen_nn::{init::Initialize, optim::{AdamW, AdamWConfig, Optimizer}, Linear, Module, MseLoss, Rnn, Tanh};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;
use rand::prelude::*;

#[derive(Module)]
pub struct FunctionModel<T: FloatDType> {
    pub rnn: Rnn<T>,
    pub fc1: Linear<T>,
    pub act: Tanh,
    pub fc2: Linear<T>,
}

impl<T: FloatDType> FunctionModel<T> {
    pub fn init(hidden_size: usize) -> lumen_core::Result<Self> {
        let initialize = Initialize::<T>::standard_uniform();
        let rnn = lumen_nn::rnn(1, hidden_size, &initialize)?;
        let fc1 = lumen_nn::linear(hidden_size, 2 * hidden_size, true, &initialize)?;
        let fc2 = lumen_nn::linear(2 * hidden_size, 1, true, &initialize)?;
        Ok(Self {
            rnn, fc1, act: Tanh::new(), fc2
        })
    }

    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // input: (batch_size, seq_len, 1)
        let (rnn_output, _state) = self.rnn.forward(input, None)?;
        let out = self.fc1.forward(&rnn_output)?;
        let out = self.act.forward(&out)?;
        let out = self.fc2.forward(&out)?;
        Ok(out)
    }
}

// 修改后的 Dataset
pub struct FunctionDataset {
    func: fn(f64) -> f64,
    num_samples: usize,
    seq_len: usize,
    min_x: f64, 
    max_x: f64, 
    is_random: bool,
}

impl FunctionDataset {
    pub fn new(func: fn(f64) -> f64, num_samples: usize, seq_len: usize, min_x: f64, max_x: f64, is_random: bool) -> Self {
        Self { func, num_samples, seq_len, min_x, max_x, is_random }
    }
}

impl Dataset<(Tensor<f64>, Tensor<f64>)> for FunctionDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, _index: usize) -> Option<(Tensor<f64>, Tensor<f64>)> {
        let mut rng = rand::rng();
        let normalized_xs: Vec<f64>;

        if self.is_random {
            let mut xs: Vec<f64> = (0..self.seq_len)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            normalized_xs = xs;
        } else {
            let step = 2.0 / (self.seq_len - 1) as f64;
            normalized_xs = (0..self.seq_len)
                .map(|i| -1.0 + step * i as f64)
                .collect();
        }

        let range = self.max_x - self.min_x;
        let ys: Vec<f64> = normalized_xs.iter()
            .map(|&nx| {
                let real_x = (nx + 1.0) * (range / 2.0) + self.min_x;
                (self.func)(real_x)
            })
            .collect();

        Some((
            Tensor::new(normalized_xs.as_slice()).unwrap(),
            Tensor::new(ys.as_slice()).unwrap(),
        ))
    }
}

pub struct FunctionBatcher;

impl Batcher<(Tensor<f64>, Tensor<f64>), (Tensor<f64>, Tensor<f64>)> for FunctionBatcher {
    type Error = lumen_core::Error;
    fn batch(&self, items: Vec<(Tensor<f64>, Tensor<f64>)>) -> Result<(Tensor<f64>, Tensor<f64>), Self::Error> {
        let mut xs = vec![];
        let mut ys = vec![];
        
        for item in items {
            xs.push(item.0);
            ys.push(item.1);
        }

        let xs = Tensor::stack(&xs, 0)?.unsqueeze(2)?; 
        let ys = Tensor::stack(&ys, 0)?.unsqueeze(2)?; 

        Ok((xs, ys))
    }
} 

type FunctionDataLoader = DataLoader<(Tensor<f64>, Tensor<f64>), (Tensor<f64>, Tensor<f64>), lumen_core::Error>;

pub fn get_dataloader(train_samples: usize, test_sample: usize, batch_size: usize) -> (FunctionDataLoader, FunctionDataLoader) {
    const MIN_FUNC_X: f64 = 0.;
    const MAX_FUNC_X: f64 = 2. * std::f64::consts::PI; // 0 ~ 2pi
    const SEQ_LEN: usize = 100;
    
    let func = f64::cos;

    let train_dataset = FunctionDataset::new(func, train_samples, SEQ_LEN, MIN_FUNC_X, MAX_FUNC_X, true);
    let train_dataloader = FunctionDataLoader::new(train_dataset, FunctionBatcher, batch_size, true);

    let test_dataset = FunctionDataset::new(func, test_sample, SEQ_LEN, MIN_FUNC_X, MAX_FUNC_X, false);
    let test_dataloader = FunctionDataLoader::new(test_dataset, FunctionBatcher, batch_size, false);

    (train_dataloader, test_dataloader)
}

fn result_main() -> anyhow::Result<()> {
    println!("Function fitting with Normalization~");

    const TRAIN_SAMPLES: usize = 2000; 
    const TEST_SAMPLES: usize = 10;
    const BATCH_SIZE: usize = 8;
    const EPOCHS: usize = 30;    
    const HIDDEN_SIZE: usize = 5;

    println!("Prepare data!");
    let (train_dataloader, test_dataloader) = get_dataloader(TRAIN_SAMPLES, TEST_SAMPLES, BATCH_SIZE);
    
    println!("Init model!");
    let model = FunctionModel::<f64>::init(HIDDEN_SIZE)?;
    let criterion = MseLoss;
    
    let mut optimizer = AdamW::new(model.params(), AdamWConfig::default())?;

    println!("Start train!");
    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0;
        let mut count = 0;

        for batch in train_dataloader.iter() {
            let (xs, ys) = batch?; // xs range is [-1, 1]
            
            let pred = model.forward(&xs)?; 
            let loss = criterion.forward(&pred, &ys)?;

            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            epoch_loss += loss.mean_all()?.to_scalar()?;
            count += 1;
        }
        
        println!("Epoch [{}/{}], Average Loss: {:.6}", epoch + 1, EPOCHS, epoch_loss / count as f64);
    }

    println!("Starting Evaluation...");
    let mut total_test_loss = 0.0;
    let mut test_count = 0;
    
    let mut plot_real_x = Vec::new();
    let mut plot_pred_y = Vec::new();

    for (i, batch) in test_dataloader.iter().enumerate() {
        let (xs, ys) = batch?; // (batch_size, seq_len, 1)
        
        let pred = model.forward(&xs)?;
        let loss = criterion.forward(&pred, &ys)?;
        
        total_test_loss += loss.mean_all()?.to_scalar()?;
        test_count += 1;

        if i == 0 {
            let norm_x_vec = xs.index(0)?.to_vec();
            let pred_y_vec = pred.index(0)?.to_vec();
            
            let range = 2. * std::f64::consts::PI - 0.;
            let min_x = 0.;
            
            for (nx, py) in norm_x_vec.iter().zip(pred_y_vec.iter()) {
                let real_x = (nx + 1.0) * (range / 2.0) + min_x;
                plot_real_x.push(real_x);
                plot_pred_y.push(*py);
            }
        }
    }

    println!("Test Set MSE Loss: {:.6}", total_test_loss / test_count as f64);

    println!("Plotting...");
    let data: Vec<(f64, f64)> = plot_real_x.into_iter()
        .zip(plot_pred_y.into_iter())
        .collect();
        
    let plot = Plot::new(data)
        .line_style(LineStyle::new().colour("red"))
        .legend("Prediction".into());
    
    let standard_data: Vec<(f64, f64)> = (0..100).map(|i| {
        let x = 0.0 + (2. * std::f64::consts::PI / 100.0) * i as f64;
        (x, x.cos())
    }).collect();
    let standard_plot = Plot::new(standard_data)
        .line_style(LineStyle::new().colour("blue"))
        .legend("Ground Truth".into());

    let view = ContinuousView::new()
        .add(standard_plot)
        .add(plot)
        .x_label("x (radians)")
        .y_label("y = cos(x)");
        
    Page::single(&view).save("image/plot.svg").unwrap();
    println!("Done! Check image/plot.svg");

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {}", e);
    }
}