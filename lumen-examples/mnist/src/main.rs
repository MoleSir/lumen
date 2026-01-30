use anyhow::Context;
use lumen_core::{FloatDType, Tensor};
use lumen_dataset::vision::{MnistDataLoader, MnistDataset};
use lumen_nn::optim::{Optimizer, SGD};
use lumen_nn::{init::Init, Linear, Module, ModuleInit};
use lumen_nn::functional as F;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {:?}", e);
    }
}
 
fn result_main() -> anyhow::Result<()> {
    // hpyer params
    const BATCH_SIZE: usize = 64;
    const LEARNING_RATE: f32 = 0.01;
    const EPOCHS: usize = 1;

    // load dataset 
    let train_dataset = MnistDataset::train(Some("../cache")).context("download train dataset")?;
    let train_loader = MnistDataLoader::from_dataset(train_dataset, BATCH_SIZE, true);

    let test_dataset = MnistDataset::test(Some("../cache")).context("download test dataset")?;
    let test_loader = MnistDataLoader::from_dataset(test_dataset, 1000, true);
    
    // create model
    let model = Net::<f32>::new().context("create model")?;
    let mut optimizer = SGD::new(model.params(), LEARNING_RATE);

    // train model
    for epoch in 0..EPOCHS {
        train(&model, &train_loader, &mut optimizer, epoch).with_context(|| format!("epoch {epoch} train"))?;
        test(&model, &test_loader).with_context(|| format!("epoch {epoch} test"))?;
    }

    // save model
    model.save_safetensors("../cache/mnist.safetensors").context("save model")?;

    // load from path
    let model = Net::from_safetensors(&(), "../cache/mnist.safetensors").context("load model")?;
    test(&model, &test_loader).context("test model")?;

    Ok(())
}


pub fn train(
    model: &Net<f32>,
    train_loader: &MnistDataLoader,
    optimizer: &mut impl Optimizer<f32>,
    epoch: usize,
) -> anyhow::Result<()> {
    for (batch_idx, batch) in train_loader.iter().enumerate() {
        let batch = batch.with_context(|| format!("parse {batch_idx} batch"))?; 
        let data = batch.images; // (batch, 28, 28)
        let target = batch.targets; // (batch, 1)

        // (batch, 28, 28) => (batch, 10)
        let output = model.forward(&data).context("model forward")?;
        let loss = F::nll_loss(&output, &target, F::LossReduction::Mean).context("nll loss")?;
        
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

    Ok(())
}

pub fn test(model: &Net<f32>, test_loader: &MnistDataLoader) -> anyhow::Result<()> {
    let mut test_loss = 0.0;
    let mut correct = 0;
    
    let _guard = lumen_core::NoGradGuard::new();
    for batch in test_loader.iter() {
        let batch = batch.context("parse batch")?;
        let data = batch.images; // (batch, 28, 28)
        let target = batch.targets; // (batch, 1)

        // (batch, 10)
        let output = model.forward(&data).context("model forward")?;
        
        // (batch, 10) and (batch, 1)
        test_loss += F::nll_loss(&output, &target, F::LossReduction::Mean)
            .context("cal loss")?
            .to_scalar()?;
        
        correct += output
            .argmax_keepdim(1).context("argmax")?
            .eq(&target).context("compare pred and target")?
            .true_count()?;
    }

    let test_loss = test_loss / test_loader.batch_count() as f32;
    let accuracy = 100.0 * correct as f32 / test_loader.dataset_len() as f32;
    
    println!(
        "\n Test set: Average loss: {}, Accuracy: {}/{} ({})", 
        test_loss, correct, test_loader.dataset_len(), accuracy
    );

    Ok(())
}

#[derive(Module)]
pub struct Net<T: FloatDType> {
    pub fc1: Linear<T>,
    pub fc2: Linear<T>,
    pub fc3: Linear<T>,
}

impl<T: FloatDType> ModuleInit<T> for Net<T> {
    type Config = ();
    type Error = anyhow::Error;

    fn init(_config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let fc1 = Linear::new(784, 512, true, init).context("init fc1")?;
        let fc2 = Linear::new(512, 256, true, init).context("init fc2")?;
        let fc3 = Linear::new(256, 10, true, init).context("init fc3")?;

        Ok(Self { fc1, fc2, fc3 })
    }
}

impl<T: FloatDType> Net<T> {
    pub fn new() -> anyhow::Result<Self> {
        Self::init(&(), None)
    }

    pub fn forward(&self, images: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // x: (batch, 28, 28)
        let (batch, height, width) = images.dims3()?;
        let x = images.reshape((batch, height * width)).context("reshape input")?;

        // (batch, 784) => (batch, 512)
        let out = self.fc1.forward(&x).context("fc1 forward")?;
        let out = F::relu(&out).context("relu")?;

        // (batch, 512) => (batch, 256)
        let out = self.fc2.forward(&out).context("fc2 forward")?;
        let out = F::relu(&out).context("relu")?;

        // (batch, 256) => (batch, 10)
        let out = self.fc3.forward(&out).context("fc3 forward")?;
        let probs = F::log_softmax(&out, 1).context("log softmax")?;

        Ok(probs)
    }
}