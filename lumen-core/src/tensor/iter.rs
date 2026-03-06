use crate::{StorageIndices, StorageRef, WithDType};

use super::Tensor;

pub struct TensorIter<'a, T> {
    indexes: StorageIndices<'a>,
    storage: StorageRef<'a, T>,
}

impl<'a, T: WithDType> Iterator for TensorIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let index = self.indexes.next()?;
        return self.storage.get(index)
    }
}

impl<T: WithDType> Tensor<T> {
    pub fn iter(&self) -> crate::Result<TensorIter<T>> {
        Ok(TensorIter {
            indexes: self.0.layout.storage_indices(),
            storage: self.storage_ref(0)?,
        })
    }
}

pub trait ResettableIterator: Iterator {
    fn reset(&mut self);
}

impl<'a, T: WithDType> ResettableIterator for TensorIter<'a, T> {
    fn reset(&mut self) {
        self.indexes.reset();
    }
}

impl<'a, T: WithDType> ExactSizeIterator for TensorIter<'a, T> {
    fn len(&self) -> usize {
        self.indexes.len()
    }
}