use std::convert::Infallible;
use crate::Dataset;

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
    type Error = Infallible;
    
    fn get(&self, index: usize) -> Result<Option<I>, Self::Error> {
        Ok(self.items.get(index).cloned())
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
