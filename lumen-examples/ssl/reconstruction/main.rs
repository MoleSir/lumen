use anyhow::Context;
use lumen_core::{FloatDType, Tensor};
use lumen_dataset::vision::{MnistDataLoader, MnistDataset};
use lumen_nn::optim::{Optimizer, SGD};
use lumen_nn::{Flatten, Linear, Module, ModuleForward, Sigmoid};
use lumen_nn::{functional as F, MseLoss, Relu};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}
 
fn result_main() -> anyhow::Result<()> {
    const BATCH_SIZE: usize = 64;
    const LEARNING_RATE: f32 = 0.01;
    const EPOCHS: usize = 3;

    // load dataset 
    let train_dataset = MnistDataset::train(Some("../cache")).context("download train dataset")?;
    let train_loader = MnistDataLoader::from_dataset(train_dataset, BATCH_SIZE, true);

    // create model
    let model = AutoEncoder::<f32>::new().context("create model")?;
    let mut optimizer = SGD::new(model.params(), LEARNING_RATE);
    let criterion = MseLoss::new(F::LossReduction::Mean);

    // train model
    for epoch in 0..EPOCHS {
        for (batch_idx, batch) in train_loader.iter().enumerate() {
            let batch = batch.with_context(|| format!("parse {batch_idx} batch"))?; 
            let data = batch.images; // (batch, 28, 28)
            let data_noisy = &data + 0.3 * data.randn_like(0.0, 1.0)?;

            let data_hat = model.forward(data_noisy).context("model forward").context("model forward")?;
            let loss = criterion.forward(&data, &data_hat)?;
            
            let grads = loss.backward().context("backward")?;
            optimizer.step(&grads)?;
    
            if batch_idx % 100 == 0 {
    
                println!(
                    "Train Epoch: {} [{}/{} ({:.2}%)]\tLoss: {}",
                    epoch, 
                    batch_idx * train_loader.batch_size(), 
                    train_loader.dataset_len(),
                    100.0 * batch_idx as f64 / train_loader.batch_count() as f64, 
                    loss.to_scalar()?
                );
            }
        }
    }

    Ok(())
}

#[derive(Module)]
pub struct AutoEncoder<T: FloatDType> {
    pub encoder: (Flatten, Linear<T>, Relu, Linear<T>), 
    pub decoder: (Linear<T>, Relu, Linear<T>, Sigmoid), 
}

impl<T: FloatDType> AutoEncoder<T> {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            encoder: (
                Flatten::batch(),
                Linear::new(28*28, 256, true, None)?,
                Relu::new(),
                Linear::new(256, 64, true, None)?,
            ),
            decoder: (
                Linear::new(64, 256, true, None)?,
                Relu::new(),
                Linear::new(256, 28*28, true, None)?,
                Sigmoid::new(),
            ),
        })
    }
    
    pub fn encoder_forward(&self, x: Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let x = self.encoder.forward(x)?;
        Ok(x)
    }

    pub fn decoder_forward(&self, x: Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let x = self.decoder.forward(x)?;
        let b = x.dim(0)?;
        Ok(x.reshape((b, 28, 28))?)
    }

    pub fn forward(&self, x: Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let z = self.encoder_forward(x).context("encoder forward")?;
        let x_hat = self.decoder_forward(z).context("decoder forward")?;
        Ok(x_hat)
    }
}
