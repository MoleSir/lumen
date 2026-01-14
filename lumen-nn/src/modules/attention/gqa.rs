use lumen_core::{FloatDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Init, Linear, ModuleInit, NnCtxError, NnError, NnResult};

#[derive(Debug, derive_new::new)]
pub struct GroupQueryAttentionConfig {
    pub hidden_size: usize,
    pub num_head: usize,
    pub num_kv_head: usize,
}

#[derive(Module)]
pub struct GroupQueryAttention<T: FloatDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    #[module(skip)]
    pub hidden_size: usize,
    #[module(skip)]
    pub head_size: usize,
    #[module(skip)]
    pub num_head: usize,
    #[module(skip)]
    pub num_kv_head: usize,
}

impl<T: FloatDType> ModuleInit<T> for GroupQueryAttention<T> {
    type Config = GroupQueryAttentionConfig;
    type Error = NnCtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let hidden_size = config.hidden_size;
        let num_head = config.num_head;
        let num_kv_head = config.num_kv_head;

        // Validation
        if hidden_size % num_head != 0 {
            return Err(NnError::HeadSizeCannotDivideByNumhead(hidden_size, num_head))?;
        }
        if num_kv_head == 0 || num_head % num_kv_head != 0 {
            return Err(NnError::HeadSizeCannotDivideByKvNumhead(hidden_size, num_kv_head))?;
        }

        let head_size = hidden_size / num_head;
        let kv_hidden_size = head_size * num_kv_head;
        
        // Use Xavier Uniform by default if not provided
        let init = init.unwrap_or_else(|| Init::xavier_uniform(T::one()));

        // Q projects to hidden_size (num_head * head_size)
        let q_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;
        // K and V project to kv_hidden_size (num_kv_head * head_size)
        let k_proj = Linear::new(hidden_size, kv_hidden_size, false, Some(init))?;
        let v_proj = Linear::new(hidden_size, kv_hidden_size, false, Some(init))?;
        // Output projects back to hidden_size
        let o_proj = Linear::new(hidden_size, hidden_size, false, Some(init))?;

        Ok(GroupQueryAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            hidden_size,
            head_size,
            num_head,
            num_kv_head,
        })
    }
}

impl<T: FloatDType> GroupQueryAttention<T> {
    /// Helper for default initialization
    #[inline]
    pub fn new(hidden_size: usize, num_head: usize, num_kv_head: usize) -> NnResult<Self> {
        Self::init(&GroupQueryAttentionConfig::new(hidden_size, num_head, num_kv_head), None)
    }

    /// Helper for custom initialization
    #[inline]
    pub fn new_with(hidden_size: usize, num_head: usize, num_kv_head: usize, init: Init<T>) -> NnResult<Self> {
        Self::init(&GroupQueryAttentionConfig::new(hidden_size, num_head, num_kv_head), Some(init))
    }

    pub fn forward(&self, hidden_state: &Tensor<T>) -> NnResult<Tensor<T>> {
        // hidden_state: (batch_size, seq_len, hidden_size)
        let (batch_size, seq_len, _hidden_size) = hidden_state.dims3()?;

        // 1. Calculate Q, K, V
        
        // Q: (batch_size, seq_len, hidden_size) -> (batch_size, num_head, seq_len, head_size)
        let q = self.q_proj.forward(hidden_state)?;
        let q = Self::reshape_head(&q, self.num_head, self.head_size)?;

        // K, V: (batch_size, seq_len, kv_hidden_size) -> (batch_size, num_kv_head, seq_len, head_size)
        let k = self.k_proj.forward(hidden_state)?; // Fixed: was q_proj
        let v = self.v_proj.forward(hidden_state)?; // Fixed: was q_proj
        let k = Self::reshape_head(&k, self.num_kv_head, self.head_size)?;
        let v = Self::reshape_head(&v, self.num_kv_head, self.head_size)?;

        // 2. Repeat K & V if necessary (GQA logic)
        // (batch_size, num_kv_head, seq_len, head_size) -> (batch_size, num_head, seq_len, head_size)
        let (k, v) = if self.num_kv_head != self.num_head {
            self.repeat_kv(&k, &v)?
        } else {
            (k, v)
        };

        // 3. Attention Weight
        // (batch_size, num_head, seq_len, head_size) @ (batch_size, num_head, head_size, seq_len) 
        // => (batch_size, num_head, seq_len, seq_len)
        // Note: keeping division by sqrt(hidden_size) to match your MHA implementation, 
        // though standard papers usually use sqrt(head_size).
        let scale = T::from_usize(self.hidden_size).sqrt();
        let attn_weight = q.matmul(&k.transpose_last()?)? / scale;

        // 4. Softmax
        // attn score: (batch_size, num_head, seq_len, seq_len) 
        let attn_score = crate::functional::softmax(&attn_weight, D::Minus1)?;

        // 5. Attention Result
        // (batch_size, num_head, seq_len, seq_len) @ (batch_size, num_head, seq_len, head_size) 
        // => (batch_size, num_head, seq_len, head_size)
        let attn_result = attn_score.matmul(&v)?;

        // 6. Reshape Back
        // (batch_size, num_head, seq_len, head_size) => 
        // (batch_size, seq_len, num_head, head_size) => 
        // (batch_size, seq_len, hidden_size)
        let attn_result = attn_result
            .transpose(1, 2)?
            .contiguous()
            .reshape((batch_size, seq_len, self.hidden_size))?;

        // 7. Output Projection
        let output = self.o_proj.forward(&attn_result)?;

        Ok(output)
    }

    /// Reshapes input tensor to separate heads.
    /// Input: (batch_size, seq_len, total_hidden_dim)
    /// Output: (batch_size, num_heads, seq_len, head_size)
    fn reshape_head(x: &Tensor<T>, num_heads: usize, head_size: usize) -> NnResult<Tensor<T>> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // (batch_size, seq_len, total_dim) => (batch_size, seq_len, num_heads, head_size)
        let x = x.reshape((batch_size, seq_len, num_heads, head_size))?;

        // (batch_size, seq_len, num_heads, head_size) => (batch_size, num_heads, seq_len, head_size)
        let x = x.transpose(1, 2)?;

        Ok(x)
    }

    fn repeat_kv(&self, k: &Tensor<T>, v: &Tensor<T>) -> NnResult<(Tensor<T>, Tensor<T>)> {
        let repeat_time = self.num_head / self.num_kv_head;
        let k = k.repeat_dim(1, repeat_time)?;
        let v = v.repeat_dim(1, repeat_time)?;

        Ok((k, v))
    }
}