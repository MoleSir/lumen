use rand::seq::SliceRandom;
use rand;
use lumen_core::{Tensor, op};
pub use super::DataSet;

pub struct DataLoader<DS: DataSet> {
    dataset: DS,
    batch_size: usize,
    shuffle: bool,
}

pub struct DataLoaderIter<'a, DS: DataSet> {
    loader: &'a DataLoader<DS>,
    cursor: usize,
    indices: Vec<usize>,
}

pub struct DataLoaderIntoIter<DS: DataSet> {
    loader: DataLoader<DS>,
    cursor: usize,
    indices: Vec<usize>,
}

impl<DS: DataSet> DataLoader<DS> {
    pub fn new(dataset: DS, batch_size: usize, shuffle: bool) -> Self {
        Self { dataset, batch_size, shuffle }
    }

    pub fn batch_count(&self) -> usize {
        self.dataset.len() / self.batch_size
    }

    pub fn iter<'a>(&'a self) -> DataLoaderIter<'a, DS> {
        let indices = self.get_iter_indices();
        DataLoaderIter {
            loader: self, 
            cursor: 0,
            indices,
        } 
    }

    fn get_iter_indices(&self) -> Vec<usize> {
        let len = self.dataset.len();
        if self.shuffle {
            (0..len).collect()
        } else {
            let mut rng = rand::rng();
            let mut indices: Vec<usize> = (0..len).collect();
            indices.shuffle(&mut rng);
            indices
        }
    }
}

impl<DS: DataSet> IntoIterator for DataLoader<DS> {
    type IntoIter = DataLoaderIntoIter<DS>;
    type Item = (Tensor, Tensor);
    fn into_iter(self) -> Self::IntoIter {
        let indices = self.get_iter_indices();
        DataLoaderIntoIter {
            loader: self, 
            cursor: 0,
            indices,
        } 
    }
}

impl<'a, DS: DataSet> Iterator for DataLoaderIter<'a, DS> {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(self.loader, &mut self.cursor, &self.indices)
    }
}

impl<DS: DataSet> Iterator for DataLoaderIntoIter<DS> {
    type Item = (Tensor, Tensor);
    fn next(&mut self) -> Option<Self::Item> {
        iter_next(&self.loader, &mut self.cursor, &self.indices)
    }
}

fn iter_next<DS: DataSet>(loader: &DataLoader<DS>, cursor: &mut usize, indices: &[usize]) -> Option<(Tensor, Tensor)> {
    let begin = *cursor;
    let end = *cursor + loader.batch_size;

    if end > indices.len() {
        None
    } else {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for index in begin..end {
            let index = indices[index];
            let (x, y) = loader.dataset.get(index);
            xs.push(x);
            ys.push(y);
        }

        *cursor += loader.batch_size;

        Some((
            op::stack(&xs).unwrap(),
            op::stack(&ys).unwrap(),
        ))
    }

}