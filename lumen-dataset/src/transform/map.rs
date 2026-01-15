use crate::Dataset;

pub trait Map {
    type Item;
    type Output;
    fn map(&self, item: Self::Item) -> Self::Output;
}

pub struct MapDataset<D, M> 
where 
    D: Dataset,
    M: Map<Item = D::Item>,
{
    dataset: D,
    map: M,
}

impl<D, M> MapDataset<D, M> 
where 
    D: Dataset,
    M: Map<Item = D::Item>,
{
    pub fn new(dataset: D, map: M,) -> Self {
        Self { dataset, map }
    }
}

impl<D, M> Dataset for MapDataset<D, M> 
where 
    D: Dataset,
    M: Map<Item = D::Item>,
{
    type Item = M::Output;

    fn get(&self, index: usize) -> Option<Self::Item> {
        let item = self.dataset.get(index);
        item.map(|item| self.map.map(item))
    }

    fn len(&self) -> usize {
        self.dataset.len()   
    }
}

