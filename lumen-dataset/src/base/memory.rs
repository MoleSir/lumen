use super::Dataset;


pub struct InMemoryDataset<I> {
    items: Vec<I>,
}

impl<I> InMemoryDataset<I> {
    pub fn new(items: Vec<I>) -> Self {
        Self { items }
    }
}

impl<I> Dataset<I> for InMemoryDataset<I> 
where 
    I: Clone + Send + Sync
{
    fn get(&self, index: usize) -> Option<I> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
