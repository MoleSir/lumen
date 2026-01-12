use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnError, NnResult};

#[derive(Module)]
pub struct MultiHeadAttention<T: FloatDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub num_head: usize,
}

#[derive(Debug, derive_new::new)]
pub struct MultiHeadAttentionConfig {
    hidden_size: usize, 
    num_head: usize
}

impl<T: FloatDType> ModuleInit<T> for MultiHeadAttention<T> {
    type Config = MultiHeadAttentionConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let hidden_size = config.hidden_size;
        let num_head = config.num_head;

        if hidden_size % num_head != 0 {
            return Err(NnError::HeadSizeCannotDivideByNumhead(hidden_size, num_head))?;
        }

        let init = init.unwrap_or_else(|| Init::xavier_uniform(T::one()));

        let q_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let k_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let v_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        let o_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;

        Ok(MultiHeadAttention { q_proj, k_proj, v_proj, o_proj, hidden_size, num_head: config.num_head })
    }
}

impl<T: FloatDType> MultiHeadAttention<T> {
    #[inline]
    pub fn new(hidden_size: usize, num_head: usize) -> NnResult<Self> {
        Self::init(&MultiHeadAttentionConfig::new(hidden_size, num_head), None)
    }

    #[inline]
    pub fn new_with(hidden_size: usize, num_head: usize, init: Init<T>) -> NnResult<Self> {
        Self::init(&MultiHeadAttentionConfig::new(hidden_size, num_head), Some(init))
    }

    pub fn forward(&self, hidden_state: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // hidden_state: (batch_size, seq_len, hidden_size)
        let (batch_size, seq_len, hidden_size) = hidden_state.dims3()?;

        // calculate q k v: (batch_size, seq_len, hidden_size)
        let q = self.q_proj.forward(hidden_state)?;
        let k = self.k_proj.forward(hidden_state)?;
        let v = self.v_proj.forward(hidden_state)?;

        // reshape to each head: (batch_size, num_head, seq_len, head_size)
        let q = self.reshape_head(&q)?;
        let k = self.reshape_head(&k)?;
        let v = self.reshape_head(&v)?;

        // attn weight
        // (batch_size, num_head, seq_len, head_size) @ (batch_size, num_head, head_size, seq_len) => (batch_size, num_head, seq_len, seq_len)
        let attn_weight = q.matmul(&k.transpose_last()?)? / T::from_usize(self.hidden_size).sqrt();

        // mask: (batch_size, num_head, seq_len, seq_len)
        let attn_weight_masked = attn_weight;

        // attn score: (batch_size, num_head, seq_len, seq_len) 
        let attn_score = crate::functional::softmax(&attn_weight_masked, D::Minus1)?;

        // attn result: (batch_size, num_head, seq_len, seq_len) @ (batch_size, num_head, seq_len, head_size) => (batch_size, num_head, seq_len, head_size)
        let attn_result = attn_score.matmul(&v)?;

        // reshape back
        // (batch_size, num_head, seq_len, head_size) => 
        // (batch_size, seq_len, num_head, head_size) => 
        // (batch_size, seq_len, hidden_size)
        let attn_result = attn_result
            .transpose(1, 2)?
            .contiguous()
            .reshape((batch_size, seq_len, hidden_size))?;

        // output: (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        let output = self.o_proj.forward(&attn_result)?;

        Ok(output)
    }

    pub fn reshape_head(&self, x: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // x: (batch_size, seq_len, hidden_size)
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        assert_eq!(hidden_size, self.hidden_size);
        let head_size = self.hidden_size / self.num_head;

        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, num_head, head_size)
        let x = x.reshape((batch_size, seq_len, self.num_head, head_size))?;
        // (batch_size, seq_len, num_head, head_size) => (batch_size, num_head, seq_len, head_size)
        let x = x.transpose(1, 2)?;

        Ok(x)
    }
}