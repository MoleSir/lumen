use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use lumen_nn::{init::Initialize, linear, optim::{Optimizer, SGD}, CrossEntropyLoss, Linear, Module, Relu};
use lumen_dataset::{common::{IrisBatcher, IrisDataset}, DataLoader};

#[derive(Module)]
pub struct Mlp<T: FloatDType> {
    pub linears: Vec<Linear<T>>,
    pub activates: Vec<Relu>,
}

impl<T: FloatDType> Mlp<T> {
    pub fn from_archs(arch: impl AsRef<[usize]>) -> lumen_core::Result<Self> {
        let arch = arch.as_ref();
        assert!(arch.len() >= 2);

        let init = Initialize::<T>::standard_normal(); 
        
        let mut linears = vec![];
        let mut activates = vec![];
        
        for i in 0..arch.len() - 1 {
            let in_dim = arch[i];
            let out_dim = arch[i+1];
            
            linears.push(linear(in_dim, out_dim, true, &init)?);
            
            if i < arch.len() - 2 {
                activates.push(Relu::new());
            }
        }
        
        Ok(Mlp { linears, activates })
    }

    pub fn forward(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        let mut x = x.clone();
        
        for (i, layer) in self.linears.iter().enumerate() {
            x = layer.forward(&x)?;
            
            if i < self.activates.len() {
                x = self.activates[i].forward(&x)?;
            }
        }

        Ok(x)
    }
}

fn result_main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = IrisDataset::load(Some("../cache"))?;
    let dataloader = DataLoader::new(dataset, IrisBatcher, 16, true);
    let model = Mlp::<f32>::from_archs([4, 10, 3])?;
    let criterion = CrossEntropyLoss;
    let mut optimizer = SGD::new(model.params(), 0.1);
    
    const EPOCHS: usize = 2500;
    for epoch in 0..EPOCHS {
        for (i, batch) in dataloader.iter().enumerate() {
            let batch = batch?;
            let pred = model.forward(&batch.features)?;
            let loss = criterion.forward(&pred, &batch.targets)?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;

            if i == 0 && epoch % 10 == 0 {
                println!("Epoch: {epoch}/{EPOCHS}, Loss: {:?}", loss.mean_all()?.to_scalar());
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprint!("Err: {}", e);
    }
}