# lumen-dataset

A flexible, type-safe, and composable data loading library.

`lumen-dataset` provides the foundational building blocks for building efficient data pipelines. It decouples data access (Dataset), data transformation (Map), and batch collation (Batcher), allowing users to easily load standard datasets or integrate custom data sources into their training loops.



## Key Features

- Composable Design: Chain datasets with lazy transformations using MapDataset.

- Flexible Batching: Custom Batcher trait allows full control over how samples are collated (e.g., stacking tensors, padding sequences).

- Standard Datasets: Built-in support for classic datasets like MNIST (Vision) and Iris (Tabular).

- Efficient Iteration: DataLoader handles shuffling, batching, and index management efficiently.

- Utilities: Helpers for random splitting, subset selection, and data conversion.



## Core Concepts

The library is built around three fundamental components that work together to create an efficient data pipeline:

### `Dataset`

```rust
pub trait Dataset {
    type Item;

    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<Self::Item>;

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, Self::Item>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}
```

The `Dataset` trait acts as the interface to your raw data. It abstracts away the storage details, providing a unified way to access individual samples.

- Responsibility: Random access to data items.
- Key Methods: get(index) and len().
- Extensibility: Can be wrapped with MapDataset for lazy transformations or SubsetDataset for subsetting (e.g., train/test splits).

### `Batcher`

```rust
pub trait Batcher {
    type Item;
    type Output;
    fn batch(&self, items: Vec<Self::Item>) -> Self::Output;
}
```

The `Batcher` trait defines how to collate a list of individual items into a single batch object (e.g., a Tensor).

- Responsibility: Converting a Vec<Item> into a specific Output type.
- Flexibility: Unlike rigid data loaders in other frameworks, the Batcher gives you full control over memory allocation, padding strategies, and tensor creation.

### `DataLoader`

```rust
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
```

The `DataLoader` is the orchestrator that drives the data loading process. It combines a specific Dataset with a specific Batcher.

- Responsibility: Handling iteration logic, shuffling, batch sizing, and index management.
- Workflow: It generates a stream of batches by selecting indices, fetching items from the Dataset, and passing them to the Batcher.

### Data Flow

```
[ Dataset ]  -->  (fetch items)  -->  [ Batcher ]  -->  (collate)  -->  [ DataLoader Yields ]
     ^                                     ^                                      ^
  Raw Data                             Aggregation                           Training Loop
(Image, Text)                        (Tensor Stacking)                        (Batch)
```



## LICENSE

MIT