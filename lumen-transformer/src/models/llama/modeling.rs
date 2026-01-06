use std::collections::HashMap;
use lumen_core::{FloatDType, Tensor};
use lumen_macros::Module;
use lumen_nn::Linear;
use super::{LlamaConfig, LlamaResult};


// ========================================================================= //
//                Attention
// ========================================================================= //

#[derive(Module)]
pub struct LlamaAttention<T: FloatDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,

    #[module(skip)]
    pub num_attention_heads: usize,
    #[module(skip)]
    pub num_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub max_position_embeddings: usize,
}

impl<T: FloatDType> LlamaAttention<T> {
    pub fn forward(&self, x: &Tensor<T>, index_pos: usize, block_idx: usize, cache: &mut LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?; // (batch_size, seq_len, head_dim * num_attn_heads)
        let k = self.k_proj.forward(x)?; // (batch_size, seq_len, head_dim * num_kv_heads)
        let v = self.v_proj.forward(x)?; // (batch_size, seq_len, head_dim * num_kv_heads)

        // (batch_size, seq_len, head_dim * num_attn_heads) => 
        // (batch_size, seq_len, num_attn_heads, head_dim) => 
        // (batch_size, num_attn_heads, seq_len, head_dim)
        let q = q.reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous();

        // (batch_size, seq_len, head_dim * num_kv_heads) => 
        // (batch_size, seq_len, num_kv_heads, head_dim) => 
        // (batch_size, num_kv_heads, seq_len, head_dim)
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous();

        // (batch_size, seq_len, head_dim * num_kv_heads) => 
        // (batch_size, seq_len, num_kv_heads, head_dim) => 
        // (batch_size, num_kv_heads, seq_len, head_dim)
        let mut v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous();

        let q = self.apply_rotary_emb(&q, index_pos, cache)?; // (batch_size, num_attn_heads, seq_len, head_dim)
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?; // (batch_size, num_kv_heads, seq_len, head_dim)
    
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                // cat [(batch_size, num_kv_heads, cache_seq_len, head_dim), (batch_size, num_kv_heads, seq_len, head_dim)]
                // => (batch_size, num_kv_heads, total_seq_len, head_dim)
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous();
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous();
                
                // if too long
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(2, k_seq_len - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous();
                }

                let v_seq_len = k.dims()[2];
                if v_seq_len > self.max_position_embeddings {
                    v = v
                        .narrow(2, v_seq_len - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous();
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        // (batch_size, num_attn_heads, total_seq_len, head_dim)
        let (k, v) = self.repeat_kv(&k, &v)?;

        // q: (batch_size, num_attn_heads, seq_len, head_dim)
        // k: (batch_size, num_attn_heads, total_seq_len, head_dim)
        // v: (batch_size, num_attn_heads, total_seq_len, head_dim)

        // (batch_size, num_attn_heads, seq_len, head_dim) @ (batch_size, num_attn_heads, total_seq_len, head_dim).T 
        // => (batch_size, num_attn_heads, seq_len, total_seq_len)
        // for each token in seq_len, get attn_weight for all total_seq_len
        let attn_weight = q.matmul(&k.transpose_last()?)? / T::from_f64((self.head_dim as f64).sqrt());
        let attn_weight_masked = if seq_len > 1 {
            /*
            
                3 x 9

                +--+--+--+--+--+--+--+--+--+
                |  |  |  |  |  |  |  |  |  |
                +--+--+--+--+--+--+--+--+--+
                |  |  |  |  |  |  |  |  |  |
                +--+--+--+--+--+--+--+--+--+
                |  |  |  |  |  |  |  |  |  |
                +--+--+--+--+--+--+--+--+--+

                masked:

                +--+--+--+--+--+--+     +--+--+--+
                |  |  |  |  |  |  |     |  |xx|xx|
                +--+--+--+--+--+--+     +--+--+--+
                |  |  |  |  |  |  |  +  |  |  |xx|
                +--+--+--+--+--+--+     +--+--+--+
                |  |  |  |  |  |  |     |  |  |  |
                +--+--+--+--+--+--+     +--+--+--+

                
            */
            // total_seq_len = cache_len + seq_len (curr)
            let total_seq_len = k.dims()[2]; 
            let causal_mask = cache.mask(seq_len)?; // (seq_len, seq_len)
            let mask = if total_seq_len > seq_len {
                let cache_len = total_seq_len - seq_len;
                let cache_mask = Tensor::falses((seq_len, cache_len))?;  // (seq_len, cache_len)
                Tensor::cat(&[&cache_mask, &causal_mask], 1)? // (seq_len, total_seq_len)
            } else {
                causal_mask
            };
            mask.if_else(<T as FloatDType>::min_value(), attn_weight)?
        } else {
            attn_weight
        }; // (batch_size, num_attn_heads, seq_len, total_seq_len)

        // (batch_size, num_attn_heads, seq_len, total_seq_len)
        let attn_score = lumen_nn::functional::softmax(&attn_weight_masked, 3)?;

        // (batch_size, num_attn_heads, seq_len, total_seq_len) @ (batch_size, num_attn_heads, total_seq_len, head_dim)
        // => (batch_size, num_attn_heads, seq_len, head_dim)
        let attn_value = attn_score.matmul(&v)?;
        // (batch_size, num_attn_heads, seq_len, head_dim) => 
        // (batch_size, seq_len, num_attn_heads, head_dim) => 
        // (batch_size, seq_len, hidden_size)
        let attn_value = attn_value.transpose(1, 2)?.reshape((batch_size, seq_len, hidden_size))?;

        // => (batch_size, seq_len, hidden_size)
        let attn_out = self.o_proj.forward(&attn_value)?;

        Ok(attn_out)
    }

    fn apply_rotary_emb(&self, x: &Tensor<T>, index_pos: usize, cache: &LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        // (batch_size, _n_heads, seq_len, head_dim)
        let (_batch_size, _n_heads, seq_len, head_dim) = x.dims4()?;

        let cos = cache.cos.narrow(0, index_pos, seq_len)?; // (seq_len, head_dim/2)
        let sin = cache.sin.narrow(0, index_pos, seq_len)?; // (seq_len, head_dim/2)
        let (cos_seq_len, cos_n_embd) = cos.dims2()?;
        let (sin_seq_len, sin_n_embd) = sin.dims2()?;

        assert_eq!(2 * cos_n_embd, head_dim);
        assert_eq!(2 * sin_n_embd, head_dim);
        assert!(seq_len <= cos_seq_len);
        assert!(seq_len <= sin_seq_len);

        let cos = Tensor::cat(&[&cos, &cos], 1)?; // (seq_len, head_dim)
        let sin = Tensor::cat(&[&sin, &sin], 1)?; // (seq_len, head_dim)
        let cos = cos.reshape((1, 1, seq_len, head_dim))?; // (1, 1, seq_len, head_dim)
        let sin = sin.reshape((1, 1, seq_len, head_dim))?; // (1, 1, seq_len, head_dim)

        // rotate_half: [-x2, x1]
        let x_rotated = self.rotate_half(x)?;

        // x: (batch_size, _n_heads, seq_len, head_dim)
        // x_rotated: (batch_size, _n_heads, seq_len, head_dim)
        // cos: (1, 1, seq_len, head_dim)
        // sin: (1, 1, seq_len, head_dim)
        let x_out = x.broadcast_mul(&cos)? + x_rotated.broadcast_mul(&sin)?;

        Ok(x_out)
    }

    fn rotate_half(&self, x: &Tensor<T>) -> LlamaResult<Tensor<T>> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dims()[last_dim];
        let half_dim = dim_size / 2;

        let x1 = x.narrow(last_dim, 0, half_dim)?;
        let x2 = x.narrow(last_dim, half_dim, half_dim)?;

        let neg_x2 = x2.neg();

        let x_rotated = Tensor::cat(&[&neg_x2, &x1], last_dim)?;
        Ok(x_rotated)
    }

    fn repeat_kv(&self, k: &Tensor<T>, v: &Tensor<T>) -> LlamaResult<(Tensor<T>, Tensor<T>)> {
        let repeat_times = self.num_attention_heads / self.num_kv_heads;
        let k = k.repeat_dim(1, repeat_times)?;
        let v = v.repeat_dim(1, repeat_times)?;
        Ok((k, v))
    }
}

pub struct LlamaCache<T: FloatDType> {
    pub use_kv_cache: bool,

    masks: HashMap<usize, Tensor<bool>>,
    kvs: Vec<Option<(Tensor<T>, Tensor<T>)>>,
    cos: Tensor<T>,
    sin: Tensor<T>,
}

impl<T: FloatDType> LlamaCache<T> {
    pub fn new(use_kv_cache: bool, config: &LlamaConfig) -> LlamaResult<Self> {
        let theta = calculate_default_inv_freq::<T>(config);
        
        //    [0, 2, 4, ...] 
        // => [ 1/rope_theta^{0/head_dim}, 1/rope_theta^{2/head_dim}, ... ]
        let theta = Tensor::new(theta)?;  // (head_dim//2, )
        let theta = theta.reshape((1, theta.element_count()))?; // (1, head_dim//2, )

        let idx_theta = Tensor::arange(T::zero(), T::from_usize(config.max_position_embeddings))?;
        let idx_theta = idx_theta.reshape((config.max_position_embeddings, 1))?; // (max_position_embeddings, 1)
        let idx_theta = idx_theta.matmul(&theta)?; // (max_position_embeddings, head_dim//2)

        let cos = idx_theta.cos(); // (max_position_embeddings, head_dim//2)
        let sin = idx_theta.sin(); // (max_position_embeddings, head_dim//2)

        Ok(Self { 
            use_kv_cache,
            masks: HashMap::new(),
            kvs: vec![None; config.num_hidden_layers],
            cos, sin,
         })
    }

    pub fn mask(&mut self, t: usize) -> LlamaResult<Tensor<bool>> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask = Tensor::triu(t, false)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}


fn calculate_default_inv_freq<T: FloatDType>(config: &LlamaConfig) -> Vec<T> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| T::one() / T::from_f64(config.rope_theta.powf(i as f64 / head_dim as f64)))
        .collect()
}

