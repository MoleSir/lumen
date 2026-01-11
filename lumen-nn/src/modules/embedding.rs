use lumen_core::{FloatDType, IntTensor, Tensor};
use lumen_macros::Module;
use crate::init::Initialize;

pub fn embedding<T: FloatDType>(num_embeddings: usize, embedding_dim: usize, init: &Initialize<T>) -> lumen_core::Result<Embedding<T>> {
    let embeddings = init.init((num_embeddings, embedding_dim))?;
    Ok(Embedding::new(embeddings, embedding_dim))
}

#[derive(Module)]
pub struct Embedding<T: FloatDType> {
    pub embeddings: Tensor<T>,
    #[module(skip)]
    pub embedding_dim: usize,
}

impl<T: FloatDType> Embedding<T> {
    pub fn new(embeddings: Tensor<T>, embedding_dim: usize) -> Self {
        Self {
            embeddings,
            embedding_dim,
        }
    }

    pub fn forward(&self, indexes: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
        let indexes = indexes.into();
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.embedding_dim);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}
