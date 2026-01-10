use rand::seq::SliceRandom;
use super::Dataset;
use super::Batcher;

pub struct DataLoader<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item>
{
    dataset: D,
    batcher: B,
    batch_size: usize,
    shuffle: bool,
} 

pub struct DataLoaderIter<'a, D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item>
{
    loader: &'a DataLoader<D, B>,
    cursor: usize,
    indices: Vec<usize>,
}

pub struct DataLoaderIntoIter<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item>
{
    loader: DataLoader<D, B>,
    cursor: usize,
    indices: Vec<usize>,
}

impl<D, B> DataLoader<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> 
{
    pub fn new(dataset: D, batcher: B, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batcher,
            batch_size,
            shuffle,
        }
    }

    pub fn batch_count(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    pub fn iter<'a>(&'a self) -> DataLoaderIter<'a, D, B> {
        let indices = self.get_iter_indices();
        DataLoaderIter { loader: self, cursor: 0, indices }
    }

    fn get_iter_indices(&self) -> Vec<usize> {
        let len = self.dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        
        if self.shuffle {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        indices
    }
}

impl<D, B> DataLoader<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> + Default
{
    pub fn from_dataset(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        Self::new(dataset, B::default(), batch_size, shuffle)
    }
}

impl<D, B> Iterator for DataLoaderIntoIter<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> 
{
    type Item = B::Output;
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(&self.loader, &mut self.cursor, &self.indices)
    }
}

impl<'a, D, B> Iterator for DataLoaderIter<'a, D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> 
{
    type Item = B::Output;
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(&self.loader, &mut self.cursor, &self.indices)
    }
}

impl<D, B> IntoIterator for DataLoader<D, B> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> 
{
    type IntoIter = DataLoaderIntoIter<D, B>;
    type Item = B::Output;
    fn into_iter(self) -> Self::IntoIter {
        let indices = self.get_iter_indices();
        DataLoaderIntoIter {
            loader: self, 
            cursor: 0,
            indices,
        } 
    }
}

fn iter_next<D, B>(loader: &DataLoader<D, B>, cursor: &mut usize, indices: &[usize]) -> Option<B::Output> 
where 
    D: Dataset,
    B: Batcher<Item = D::Item> 
{
    let begin = *cursor;
    if begin >= loader.dataset.len() {
        None
    } else {
        let end = *cursor + loader.batch_size;
        let end = end.min(loader.dataset.len());
        let mut items = vec![];
        for index in begin..end {
            let index = indices[index];
            items.push(loader.dataset.get(index).unwrap());
        }        
        *cursor += items.len();
        let batch = loader.batcher.batch(items);
        Some(batch)
    }
}