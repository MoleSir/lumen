use lumen_dataset::Dataset;
use lumen_macros::Module;

#[derive(Module)]
pub struct FunctionModel {
    
}

struct FunctionDataset {
    func: fn(f64) -> f64,
    num_samples: usize,
    min_x: f64,
    max_x: f64,
}

impl FunctionDataset {
    fn new(func: fn(f64) -> f64, num_samples: usize, min_x: f64, max_x: f64) -> Self {
        Self { func, num_samples, min_x, max_x }
    }
}

impl Dataset<(f64, f64)> for FunctionDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Option<(f64, f64)> {
        unimplemented!()
    }
}   



fn result_main() -> anyhow::Result<()> {
    Ok(())
}

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {}", e);
    }
}