use lumen_core::{FloatDType, IntTensor, Tensor};
use lumen_macros::Module;
use crate::{init::Init, NnCtxError, NnResult};
use super::ModuleInit;

/// A simple lookup table that stores embeddings of a fixed dictionary and size.
#[derive(Module)]
pub struct Embedding<T: FloatDType> {
    pub embeddings: Tensor<T>,

    #[module(skip)]
    pub num_embeddings: usize,
    #[module(skip)]
    pub embedding_size: usize,
}

#[derive(Debug, Clone, derive_new::new)]
pub struct EmbeddingConfig {
    pub num_embeddings: usize,
    pub embedding_size: usize,
}

impl<T: FloatDType> ModuleInit<T> for Embedding<T> {
    type Config = EmbeddingConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let init = init.unwrap_or(Init::standard_normal());
        let embeddings = init.init((config.num_embeddings, config.embedding_size))?;
        Ok(Self { embeddings, num_embeddings: config.num_embeddings, embedding_size: config.embedding_size })
    }
}

impl<T: FloatDType> Embedding<T> {
    #[inline]
    pub fn new(num_embeddings: usize, embedding_size: usize, init: Option<Init<T>>) -> NnResult<Self> {
        Self::init(&EmbeddingConfig::new(num_embeddings, embedding_size), init)
    }

    pub fn forward(&self, indexes: impl Into<IntTensor>) -> lumen_core::Result<Tensor<T>> {
        let indexes = indexes.into();
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.embedding_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}
