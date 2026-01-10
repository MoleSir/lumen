mod dataloader;
mod batcher;
mod error;
mod utils;
mod datasets;
pub mod transform;
pub use dataloader::*;
pub use batcher::*;
pub use error::*;
pub use datasets::*;

pub trait Dataset {
    type Item;

    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, Self::Item>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

//===========================================================//
//                Iter
//===========================================================//

pub struct DatasetIterator<'a, I> {
    current: usize,
    dataset: &'a dyn Dataset<Item = I>,
}

impl<'a, I> DatasetIterator<'a, I> {
    /// Creates a new dataset iterator.
    pub fn new<D>(dataset: &'a D) -> Self
    where
        D: Dataset<Item = I>,
    {
        DatasetIterator {
            current: 0,
            dataset,
        }
    }
}

impl<I> Iterator for DatasetIterator<'_, I> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        let item = self.dataset.get(self.current);
        self.current += 1;
        item
    }
}

//===========================================================//
//                Vec Dataset
//===========================================================//

pub struct VecDataset<I> {
    items: Vec<I>,
}

impl<I> VecDataset<I> {
    pub fn new(items: Vec<I>) -> Self {
        Self { items }
    }
}

impl<I> Dataset for VecDataset<I> 
where 
    I: Clone
{
    type Item = I;
    
    fn get(&self, index: usize) -> Option<I> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
