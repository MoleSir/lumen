use lumen_core::{FloatDType, NumDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Initialize, linear, Linear};

pub fn self_attention<T: FloatDType>(hidden_size: usize, init: &Initialize<T>) -> lumen_core::Result<SelfAttention<T>> {
    let q_proj = linear(hidden_size, hidden_size, false, init)?;
    let k_proj = linear(hidden_size, hidden_size, false, init)?;
    let v_proj = linear(hidden_size, hidden_size, false, init)?;
    let o_proj = linear(hidden_size, hidden_size, false, init)?;
    Ok(SelfAttention { q_proj, k_proj, v_proj, hidden_size, o_proj })
}

#[derive(Module)]
pub struct SelfAttention<T: NumDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    #[module(skip)]
    pub hidden_size: usize,
}

impl<T: FloatDType> SelfAttention<T> {
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