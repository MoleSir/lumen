pub mod dataloader;
pub use dataloader::DataLoader;
use lumen_core::Tensor;

pub trait DataSet {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (Tensor, Tensor);
}