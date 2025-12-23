use lumen_core::{FloatDType, NumDType, Tensor};
use crate::init::Initialize;
use super::{Module, ModuleVisitor};

pub fn embedding<T: FloatDType>(num_embeddings: usize, embedding_dim: usize, init: &Initialize<T>) -> lumen_core::Result<Embedding<T>> {
    let embeddings = init.init((num_embeddings, embedding_dim))?;
    Ok(Embedding::new(embeddings, embedding_dim))
}

pub struct Embedding<T: NumDType> {
    pub embeddings: Tensor<T>,
    pub embedding_dim: usize,
}

impl<T: NumDType> Embedding<T> {
    pub fn new(embeddings: Tensor<T>, embedding_dim: usize) -> Self {
        Self {
            embeddings,
            embedding_dim,
        }
    }

    pub fn forward(&self, indexes: &Tensor<usize>) -> lumen_core::Result<Tensor<T>> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.embedding_dim);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

impl<T: NumDType> Module<T> for Embedding<T> {
    fn visit<Visitor: ModuleVisitor<T>>(&self, visitor: &mut Visitor) {
        visitor.enter_module("embeddings");
        visitor.visit(&self.embeddings);
        visitor.exit_module("embeddings");
    }
}