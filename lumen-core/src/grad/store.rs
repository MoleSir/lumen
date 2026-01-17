use std::{collections::HashMap, ops::Index};
use crate::{FloatDType, Tensor, TensorId};

#[derive(Debug, Clone)]
pub struct GradStore<T: FloatDType>(HashMap<TensorId, Tensor<T>>);

impl<T: FloatDType> GradStore<T> {
    /// Create a new gradient store
    pub fn new() -> Self {
        GradStore(HashMap::new())
    }

    /// Get the gradient tensor corresponding to the given tensor id
    pub fn get_id(&self, id: TensorId) -> Option<&Tensor<T>> {
        self.0.get(&id)
    }

    /// Get the gradient tensor associated with the given tensor
    pub fn get(&self, tensor: &Tensor<T>) -> Option<&Tensor<T>> {
        self.0.get(&tensor.id())
    }

    /// Remove the gradient tensor associated with the given tensor, returning it if it exists
    pub fn remove(&mut self, tensor: &Tensor<T>) -> Option<Tensor<T>> {
        self.0.remove(&tensor.id())
    }

    /// Insert a gradient tensor associated with the given tensor, returning the previous gradient tensor if it existed
    pub fn insert(&mut self, tensor: &Tensor<T>, grad: Tensor<T>) -> Option<Tensor<T>> {
        self.0.insert(tensor.id(), grad)
    }

    /// Get the gradient tensor associated with the given tensor, or, if it does not exist,
    /// insert a tensor of zeroes, with the same shape and type as the given tensors and return it
    pub fn or_insert(&mut self, tensor: &Tensor<T>) -> crate::Result<&mut Tensor<T>> {
        use std::collections::hash_map::Entry;
        let grad = match self.0.entry(tensor.id()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let grad = tensor.zeros_like()?;
                entry.insert(grad)
            }
        };
        Ok(grad)
    }

    /// Get the tensor ids of the stored gradient tensors
    pub fn get_ids(&self) -> impl Iterator<Item = &TensorId> {
        self.0.keys()
    }

    pub fn tensors(&self) -> impl Iterator<Item = &Tensor<T>> {
        self.0.values()
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, TensorId, Tensor<T>> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: FloatDType> Index<&Tensor<T>> for GradStore<T> {
    type Output = Tensor<T>;
    fn index(&self, index: &Tensor<T>) -> &Self::Output {
        self.get(index).unwrap()
    }
}