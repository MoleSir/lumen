use std::sync::Arc;
use rand::seq::SliceRandom;
use crate::{Dataset, DatasetError, DatasetResult};

pub struct SubsetDataset<D> 
where 
    D: Dataset,
{
    pub wrapped: Arc<D>,
    pub indices: Vec<usize>,
}

impl<D: Dataset> SubsetDataset<D> {
    /// Creates a new selection dataset with the given dataset and indices without checking bounds.
    ///
    /// ## Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `indices` - A vector of indices to select from the dataset.
    pub fn new(dataset: impl Into<Arc<D>>, indices: Vec<usize>) -> Self {
        Self { wrapped: dataset.into(), indices }
    }

    /// Creates a new selection dataset with the given dataset and indices.
    ///
    /// Checks that all indices are within the bounds of the dataset.
    ///
    /// ## Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    /// * `indices` - A slice of indices to select from the dataset.
    ///   These indices must be within the bounds of the dataset.
    pub fn from_indices(dataset: impl Into<Arc<D>>, indices: Vec<usize>) -> DatasetResult<Self> {
        let dataset = dataset.into();
        
        // check index
        let size = dataset.len();
        if let Some(&idx) = indices.iter().find(|&i| *i >= size) {
            Err(DatasetError::IndexOutOfRangeWhenSelectDataset(idx, size, "from indices"))?;
        }

        Ok(Self::new(dataset, indices))
    }

    /// Creates a new selection dataset that selects all indices from the dataset.
    ///
    /// This allocates a 1-to-1 mapping of indices to the dataset size,
    /// essentially functioning as a no-op selection. This is only useful
    /// when the dataset will later be shuffled or transformed in place.
    ///
    /// ## Arguments
    ///
    /// * `dataset` - The original dataset to select from.
    ///
    /// ## Returns
    ///
    /// A new `SubsetDataset` that selects all indices from the dataset.
    pub fn select_all(dataset: impl Into<Arc<D>>) -> Self {
        let dataset = dataset.into();
        let size = dataset.len();
        Self::new(dataset, iota(size))
    }

    /// Creates a new dataset that is a slice of the current selection dataset.
    ///
    /// Slices the *selection indices* from ``[start..end]``.
    ///
    /// Independent of future shuffles on the parent, but shares the same wrapped dataset.
    ///
    /// ## Arguments
    ///
    /// * `start` - The start of the range.
    /// * `end` - The end of the range (exclusive).
    pub fn slice(&self, start: usize, end: usize) -> DatasetResult<Self> {
        if start >= self.len() {
            Err(DatasetError::IndexOutOfRangeWhenSelectDataset(start, self.len(), "slice"))?;
        }
        if end >= self.len() {
            Err(DatasetError::IndexOutOfRangeWhenSelectDataset(end, self.len(), "slice"))?;
        }
        Ok(Self::new(self.wrapped.clone(), self.indices[start..end].to_vec()))
    }

    /// Split into `num` datasets by slicing the selection indices evenly.
    ///
    /// Split is done via `slice`, so the datasets share the same wrapped dataset.
    ///
    /// Independent of future shuffles on the parent, but shares the same wrapped dataset.
    ///
    /// ## Arguments
    ///
    /// * `num` - The number of datasets to split into.
    ///
    /// ## Returns
    ///
    /// A vector of `SubsetDataset` instances, each containing a subset of the indices.
    pub fn split(&self, num: usize) -> DatasetResult<Vec<Self>> {
        if num == 0 {
            Err(DatasetError::NumSplitZeroWhenSelectDataset)?;
        }
        
        let n = self.indices.len();
        let mut datasets = Vec::with_capacity(num);
        
        let base_size = n / num;
        let remainder = n % num;

        let mut start = 0;
        for i in 0..num {
            let size = base_size + (if i < remainder { 1 } else { 0 });
            let end = start + size;
            
            datasets.push(self.slice(start, end)?);
            
            start = end;
        }

        Ok(datasets)
    }
}

impl<D: Dataset> Dataset for SubsetDataset<D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Option<Self::Item> {
        let index = self.indices.get(index)?;
        self.wrapped.get(*index)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

pub fn random_split<D: Dataset>(dataset: D, ratio: f64) -> (SubsetDataset<D>, SubsetDataset<D>) {
    let length = dataset.len();
    let mut indices = iota(length);
    
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);

    let split_idx = (length as f64 * ratio).floor() as usize;
    let (indices1, indices2) = indices.split_at(split_idx);

    let dataset = Arc::new(dataset);
    let subset1 = SubsetDataset::new(dataset.clone(), indices1.to_vec());
    let subset2 = SubsetDataset::new(dataset.clone(), indices2.to_vec());

    (subset1, subset2)
}

#[inline(always)]
fn iota(size: usize) -> Vec<usize> {
    (0..size).collect()
}

