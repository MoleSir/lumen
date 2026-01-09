use rand::seq::SliceRandom;
use super::Dataset;

pub trait Batcher<I, O>: Send + Sync {
    type Error;
    fn batch(&self, items: Vec<I>) -> Result<O, Self::Error>;
}

pub struct DataLoader<I, O, E> {
    dataset: Box<dyn Dataset<I>>,
    batcher: Box<dyn Batcher<I, O, Error = E>>, 
    batch_size: usize,
    shuffle: bool,
}

pub struct DataLoaderIter<'a, I, O, E> {
    loader: &'a DataLoader<I, O, E>,
    cursor: usize,
    indices: Vec<usize>,
}

pub struct DataLoaderIntoIter<I, O, E> {
    loader: DataLoader<I, O, E>,
    cursor: usize,
    indices: Vec<usize>,
}

impl<I, O, E> DataLoader<I, O, E> {
    pub fn new(
        dataset: impl Dataset<I> + 'static,
        batcher: impl Batcher<I, O, Error = E> + 'static, 
        batch_size: usize,
        shuffle: bool,
    ) -> Self {
        Self {
            dataset: Box::new(dataset),
            batcher: Box::new(batcher),
            batch_size,
            shuffle
        }
    }

    pub fn batch_count(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    pub fn iter<'a>(&'a self) -> DataLoaderIter<'a, I, O, E> {
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

impl<I, O, E> Iterator for DataLoaderIntoIter<I, O, E> {
    type Item = Result<O, E>;
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(&self.loader, &mut self.cursor, &self.indices)
    }
}

impl<'a, I, O, E> Iterator for DataLoaderIter<'a, I, O, E> {
    type Item = Result<O, E>;
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(&self.loader, &mut self.cursor, &self.indices)
    }
}

impl<I, O, E> IntoIterator for DataLoader<I, O, E> {
    type IntoIter = DataLoaderIntoIter<I, O, E>;
    type Item = Result<O, E>;
    fn into_iter(self) -> Self::IntoIter {
        let indices = self.get_iter_indices();
        DataLoaderIntoIter {
            loader: self, 
            cursor: 0,
            indices,
        } 
    }
}

fn iter_next<I, O, E>(loader: &DataLoader<I, O, E>, cursor: &mut usize, indices: &[usize]) -> Option<Result<O, E>> {
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