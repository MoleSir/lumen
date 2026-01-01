use std::marker::PhantomData;
use crate::Dataset;


pub trait Map<I, O>: Send + Sync {
    fn map(&self, item: &I) -> O;
}

pub struct MapDataset<D, M, I> {
    dataset: D,
    map: M,
    input: PhantomData<I>,
}

impl<D, M, I, O> Dataset<O> for MapDataset<D, M, I> 
where 
    D: Dataset<I>,
    M: Map<I, O>,
    I: Send + Sync,
    O: Send + Sync
{
    fn get(&self, index: usize) -> Option<O> {
        let item = self.dataset.get(index);
        item.map(|item| self.map.map(&item))
    }

    fn len(&self) -> usize {
        self.dataset.len()   
    }
}

