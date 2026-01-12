use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnResult};

#[derive(Module)]
pub struct SelfAttention<T: FloatDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    #[module(skip)]
    pub hidden_size: usize,
}

impl<T: FloatDType> ModuleInit<T> for SelfAttention<T> {
    type Config = usize;
    type Error = NnCtxError;

    fn init(hidden_size: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let hidden_size = *hidden_size;
        let init = init.unwrap_or_else(|| Init::xavier_uniform(T::one()));
        let q_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let k_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let v_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let o_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        Ok(SelfAttention { q_proj, k_proj, v_proj, hidden_size, o_proj })
    }
}

impl<T: FloatDType> SelfAttention<T> {
    #[inline]
    pub fn new(hidden_size: usize) -> NnResult<Self> {
        Self::init_default(&hidden_size)
    }

    #[inline]
    pub fn new_with(hidden_size: usize, init: Init<T>) -> NnResult<Self> {
        Self::init_with(&hidden_size, init)
    }

    pub fn forward(&self, hidden_state: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // calculate q k v 
        let q = self.q_proj.forward(hidden_state)?;
        let k = self.k_proj.forward(hidden_state)?;
        let v = self.v_proj.forward(hidden_state)?;

        // attn weight 
        let attn_weight = q.matmul(&k.transpose_last()?)? / T::from_usize(self.hidden_size).sqrt();

        // attn score
        let attn_score = crate::functional::softmax(&attn_weight, D::Minus1)?;

        // attn result
        let attn_result = attn_score.matmul(&v)?;
        
        // output 
        let output = self.o_proj.forward(&attn_result)?;

        Ok(output)
    }
}