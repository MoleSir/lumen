use lumen_core::{FloatDType, NumDType, Tensor, D};
use lumen_macros::Module;
use crate::{init::Initialize, linear, Linear};

pub fn group_query_attention<T: FloatDType>(
    hidden_size: usize, 
    num_head: usize, 
    num_kv_head: usize,
    init: &Initialize<T>
) -> lumen_core::Result<GroupQueryAttention<T>> {
    // TODO: error
    assert!(hidden_size % num_head == 0);
    assert!(num_head > num_kv_head &&  num_head % num_kv_head == 0);

    let head_size = hidden_size / num_head;
    let kv_hidden_size = head_size * num_kv_head;

    let q_proj = linear(hidden_size, hidden_size, false, init)?;
    let k_proj = linear(hidden_size, kv_hidden_size, false, init)?;
    let v_proj = linear(hidden_size, kv_hidden_size, false, init)?;
    let o_proj = linear(hidden_size, hidden_size, false, init)?;
    Ok(GroupQueryAttention { q_proj, k_proj, v_proj, o_proj, hidden_size, head_size, num_head, num_kv_head })
}

#[derive(Module)]
pub struct GroupQueryAttention<T: NumDType> {
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

impl<T: FloatDType> GroupQueryAttention<T> {
    pub fn forward(&self, hidden_state: &Tensor<T>) -> lumen_core::Result<Tensor<T>> {
        // hidden_state: (batch_size, seq_len, hidden_size)
        let (batch_size, seq_len, hidden_size) = hidden_state.dims3()?;

        // calculate q k v:
        // (batch_size, seq_len, hidden_size) => (batch_size, num_head, seq_len, head_size)
        let q = self.q_proj.forward(hidden_state)?;
        let q = Self::reshape_head(&q, self.num_head)?;

        // (batch_size, seq_len, kv_hidden_size) => (batch_size, num_kv_head, seq_len, head_size)
        let k = self.q_proj.forward(hidden_state)?;
        let v = self.q_proj.forward(hidden_state)?;
        let k = Self::reshape_head(&k, self.num_kv_head)?;
        let v = Self::reshape_head(&v, self.num_kv_head)?;

        // repeat k & v: (batch_size, num_kv_head, seq_len, head_size) => (batch_size, num_head, seq_len, head_size)
        let (k, v) = self.repeat_kv(&k, &v)?;

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

    pub fn reshape_head(x: &Tensor<T>, num_head: usize) -> lumen_core::Result<Tensor<T>> {
        // x: (batch_size, seq_len, hidden_size)
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        assert!(hidden_size % num_head == 0);
        let head_size = hidden_size / num_head;

        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, num_head, head_size)
        let x = x.reshape((batch_size, seq_len, num_head, head_size))?;
        // (batch_size, seq_len, num_head, head_size) => (batch_size, num_head, seq_len, head_size)
        let x = x.transpose(1, 2)?;

        Ok(x)
    }

    pub fn repeat_kv(&self, k: &Tensor<T>, v: &Tensor<T>) -> lumen_core::Result<(Tensor<T>, Tensor<T>)> {
        assert!(self.num_head % self.num_kv_head == 0);
        let repeat_time = self.num_head / self.num_kv_head;

        let k = k.repeat_dim(1, repeat_time)?;
        let v = v.repeat_dim(1, repeat_time)?;

        Ok((k, v))
    }
}