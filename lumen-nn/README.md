# lumen-nn

A modular, extensible, and type-safe deep learning framework for Rust.

`lumen-nn` provides a PyTorch-like API designed for ergonomics and performance. It leverages Rust's trait system and procedural macros to automate parameter management, initialization, and serialization, allowing researchers and engineers to focus on model architecture rather than boilerplate code.



## Key Features

- Intuitive Module Interface: Inspired by PyTorch, but built for Rust.

- Automatic Parameter Discovery: Uses a Visitor pattern and macros to automatically handle parameter registration, counting, and device movement. No need to manually register parameters.

- Native safetensors Support: First-class support for loading and saving weights using the HuggingFace safetensors format.

- Component Library:
  - Layers: Linear, Embedding, Dropout, Norms (Layer/RMS/Batch), and more.
  - Attention: Self Attention, Multi-Head, Group Query (GQA).
  - RNNs: RNN, GRU, LSTM.
  - Geometric: GCN Convolution.

- Flexible Initialization: Supports Kaiming, Xavier (Glorot), Normal, Uniform, and Constant initializations out of the box.



## Architecture

At the core of lumen-nn is the Module trait. Unlike other libraries that require manual tracking of tensors, lumen-nn uses a recursive Visitor Pattern.

When you derive or implement `Module`, the library automatically generates methods for:

- `named_params()` / `named_buffers()`
- `apply()` (for custom initializations or modifications)
- Parameter counting and serialization.



## Quick Start

### 1. Define a Model

Use the `#[derive(Module)]` macro to define your network structure effortlessly.

```rust
use lumen_nn::{Module, ModuleInit, Linear, Init, FloatDType};
use anyhow::{Context, Result};

#[derive(Module, Clone)]
pub struct MLP<T: FloatDType> {
    pub fc1: Linear<T>,
    pub fc2: Linear<T>,
    pub fc3: Linear<T>,
}

impl<T: FloatDType> ModuleInit<T> for MLP<T> {
    type Config = (); // Custom config struct can be used here
    type Error = anyhow::Error;

    fn init(_config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        // Initialize layers with optional weight initialization strategies
        let fc1 = Linear::new(784, 512, true, init).context("init fc1")?;
        let fc2 = Linear::new(512, 256, true, init).context("init fc2")?;
        let fc3 = Linear::new(256, 10, true, init).context("init fc3")?;

        Ok(Self { fc1, fc2, fc3 })
    }
}
```

### 2. Training & Serialization

`lumen-nn` integrates seamlessly with safetensors for model persistence.

````rust
fn main() -> anyhow::Result<()> {
    // Create Dataset
    let train_dataset = MnistDataset::train(Some("../cache")).context("download train dataset")?;

    // Initialize Model
    let mut model = MLP::<f32>::new().context("failed to create model")?;
    
    // Setup Optimizer
    let mut optimizer = lumen_nn::optim::SGD::new(model.params(), 0.01);

    // Training Loop (Simplified)
    for epoch in 0..10 {
        // For batch
        for (batch_idx, batch) in train_loader.iter().enumerate() {
            let batch = batch.with_context(|| format!("parse {batch_idx} batch"))?; 
            let data = batch.images;
            let target = batch.targets;

            // Front forward
            let output = model.forward(&data).context("model forward")?;
            // Get loss
            let loss = F::nll_loss(&output, &target).context("nll loss")?;

            // Backward to get grads
            let grads = loss.backward().context("backward")?;
            
            // Use optimizer to update weight
            optimizer.step(&grads)?;
        }
    }

    // Save Weights
    model.save_safetensors("./checkpoints/mnist_mlp.safetensors")?;

    // Load Weights
    let loaded_model = MLP::<f32>::from_safetensors(&(), "./checkpoints/mnist_mlp.safetensors")?;
    
    Ok(())
}
````



## Supported Modules

| Category	| Modules |
| :---:  | --- |
| Common |	`Linear`, `Embedding`, `Dropout` |
| Attention |	`SelfAttention`, `MultiHeadSelfAttention`, `GroupQuerySelfAttention` |
| RNN |	`Rnn`, `Gru`, `Lstm` |
| Norm	| `BatchNorm1D`, `LayerNorm`, `RMSNorm` |
| Activation	| `Gelu`, `LeakyRelu`, `Relu`, `Sigmoid`, `Softmax`, `Tanh` |
| Loss	| `MseLoss`, `CrossEntropyLoss` |
| Graph	| `GCNConv` |



## LICENSE

MIT