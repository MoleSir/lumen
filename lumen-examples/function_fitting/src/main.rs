use lumen_core::{FloatDType, Tensor};
use lumen_dataset::{Batcher, DataLoader, Dataset};
use lumen_nn::{init::Initialize, optim::{AdamW, AdamWConfig, Optimizer}, Linear, Module, MseLoss, Rnn, Tanh};
use plotlib::page::Page;
use plotlib::repr::Plot;
use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;

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
        // Input dim is 1 (x value)
        let rnn = lumen_nn::rnn(1, hidden_size, &initialize)?;
        let fc1 = lumen_nn::linear(hidden_size, 2 * hidden_size, true, &initialize)?;
        let fc2 = lumen_nn::linear(2 * hidden_size, 1, true, &initialize)?;
        Ok(Self {
            rnn, fc1, act: Tanh::new(), fc2
        })
    }

    pub fn forward(&self, input: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // input: (batch_size, seq_len, 1)
        
        // RNN Output: (batch_size, seq_len, hidden_size)
        let (rnn_output, _state) = self.rnn.forward(input, None)?;

        // FC1: (batch_size, seq_len, 2 * hidden_size)
        // Note: Removed the extra .tanh() call here because self.act does it next
        let out = self.fc1.forward(&rnn_output)?;
        
        // Activation: (batch_size, seq_len, 2 * hidden_size)
        let out = self.act.forward(&out)?;
        
        // FC2: (batch_size, seq_len, 1)
        let out = self.fc2.forward(&out)?;

        Ok(out)
    }
}

pub struct FunctionDataset {
    xs: Vec<f64>,
    ys: Vec<f64>,
    num_samples: usize,
}

impl FunctionDataset {
    pub fn new(func: fn(f64) -> f64, num_samples: usize, seq_len: usize, min_x: f64, max_x: f64) -> Self {
        let step = (max_x - min_x) / seq_len as f64;
        let xs = (0..seq_len)
            .map(|i| min_x + step * i as f64 )
            .collect();
        let ys = (0..seq_len)
            .map(|i| func(min_x + step * i as f64) )
            .collect();
        Self { xs, ys, num_samples }
    }
}

impl Dataset<(Tensor<f64>, Tensor<f64>)> for FunctionDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, _index: usize) -> Option<(Tensor<f64>, Tensor<f64>)> {
        Some((
            Tensor::new(self.xs.as_slice()).unwrap(),
            Tensor::new(self.ys.as_slice()).unwrap(),
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

        let xs = Tensor::stack(&xs, 0)?.unsqueeze(2)?; // (batch_size, seq_len=1, input_dim=1)
        let ys = Tensor::stack(&ys, 0)?.unsqueeze(2)?; // (batch_size, seq_len=1, output_dim=1)

        Ok((xs, ys))
    }
} 

type FunctionDataLoader = DataLoader<(Tensor<f64>, Tensor<f64>), (Tensor<f64>, Tensor<f64>), lumen_core::Error>;

pub fn get_dataloader(train_samples: usize, test_sample: usize, batch_size: usize) -> (FunctionDataLoader, FunctionDataLoader) {
    const MIN_FUNC_X: f64 = 0.;
    const MAX_FUNC_X: f64 = 2. * std::f64::consts::PI;
    const SEQ_LEN: usize = 100;
    
    let func = f64::cos;

    let train_dataset = FunctionDataset::new(func, train_samples, SEQ_LEN, MIN_FUNC_X, MAX_FUNC_X);
    // Enable shuffle for training
    let train_dataloader = FunctionDataLoader::new(train_dataset, FunctionBatcher, batch_size, true);

    let test_dataset = FunctionDataset::new(func, test_sample, SEQ_LEN, MIN_FUNC_X, MAX_FUNC_X);
    let test_dataloader = FunctionDataLoader::new(test_dataset, FunctionBatcher, batch_size, false);

    (train_dataloader, test_dataloader)
}

fn result_main() -> anyhow::Result<()> {
    println!("Function fitting~");
    const TRAIN_SAMPLES: usize = 9000;
    const TEST_SAMPLES: usize = 1;
    const BATCH_SIZE: usize = 1;
    const EPOCHS: usize = 1;
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

        for (i, batch) in train_dataloader.iter().enumerate() {
            let (xs, ys) = batch?;
            
            // Forward pass
            let pred = model.forward(&xs)?; 
            
            // Calculate loss
            let loss = criterion.forward(&pred, &ys)?;

            // Backward and Optimize
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            // Accumulate loss for logging
            let loss_mean = loss.mean_all()?.to_scalar()?;
            epoch_loss += loss_mean;
            count += 1;

            if i % 100 == 0 {
                print!("Iter {} Loss: {:?}\n", i, loss_mean);
            }
        }
        
        println!("Epoch [{}/{}], Average Loss: {:.6}", epoch + 1, EPOCHS, epoch_loss / count as f64);
    }

    let (x, _) = test_dataloader.iter().next().unwrap()?;
    let pred = model.forward(&x)?;
    let data = x.iter().zip(pred.iter()).map(|(a, b)| (a, b)).collect();
    let plot = Plot::new(data).line_style(LineStyle::new().colour("red"));
    let view = ContinuousView::new().add(plot);
    Page::single(&view).save("image/plot.svg").unwrap();


    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {}", e);
    }
}