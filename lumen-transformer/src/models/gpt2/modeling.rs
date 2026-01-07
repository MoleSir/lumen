use std::collections::HashMap;
use lumen_core::{FloatDType, Tensor};
use super::{Gpt2Config, Gpt2Result};



// ========================================================================= //
//                Cache
// ========================================================================= //

pub struct Gpt2Cache<T: FloatDType> {
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor<T>, Tensor<T>)>>,
    masks: HashMap<usize, Tensor<bool>>,
}

impl<T: FloatDType> Gpt2Cache<T> {
    pub fn new(use_kv_cache: bool, config: &Gpt2Config) -> Self {
        Self {
            use_kv_cache,
            kvs: vec![None; config.n_layer],
            masks: HashMap::new(),
        }
    }

    pub fn mask(&mut self, t: usize) -> Gpt2Result<Tensor<bool>> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask = Tensor::triu(t, false)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}