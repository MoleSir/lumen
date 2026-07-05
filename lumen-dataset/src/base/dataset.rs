use std::fmt::Display;

pub trait Dataset {
    type Item;
    type Error: Display;

    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Result<Option<Self::Item>, Self::Error>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, Self>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

//===========================================================//
//                Iter
//===========================================================//

pub struct DatasetIterator<'a, D: Dataset> {
    index: usize,
    dataset: &'a D,
}

impl<'a, D: Dataset> DatasetIterator<'a, D> {
    /// Creates a new dataset iterator.
    pub fn new(dataset: &'a D) -> Self {
        DatasetIterator {
            index: 0,
            dataset,
        }
    }
}

impl<D: Dataset> Iterator for DatasetIterator<'_, D> {
    type Item = Result<D::Item, D::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.dataset.get(self.index) {
            Ok(Some(item)) => {
                self.index += 1;
                Some(Ok(item)) 
            }
            Ok(None) => None, 
            Err(e) => {
                self.index += 1; 
                Some(Err(e))
            }
        }
    }
}
