use std::collections::HashMap;
use lumen_core::{FloatDType, IntTensor, Tensor, Var, D};
use lumen_macros::Module;
use lumen_nn::{init::Init, Embedding, Linear, ModuleInit};
use thiserrorctx::Context;
use super::{LlamaConfig, LlamaCtxError, LlamaError, LlamaResult};

// ========================================================================= //
//                For Causal LM
// ========================================================================= //

#[derive(Module)]
pub struct LlamaForCausalLM<T: FloatDType> {
    pub model: LlamaModel<T>,
    pub lm_head: Linear<T>, 
}

impl<T: FloatDType> ModuleInit<T> for LlamaForCausalLM<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        let model = LlamaModel::init(config, init).context("init llama model")?;
        let lm_head_init = init.unwrap_or_else(default_init_linear);
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false, Some(lm_head_init))
            .map_err(LlamaError::Nn)
            .context("init lm head")?;
        
        Ok(Self { model, lm_head })
    }
}

impl<T: FloatDType> LlamaForCausalLM<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        // (batch_size, seq_len) => (batch_size, seq_len, hidden_size)
        let hidden_states = self.model.forward(input_ids, start_pos, cache).context("model forward")?;
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, vocab_size)
        let hidden_states = self.lm_head.forward(&hidden_states)
            .map_err(LlamaError::Core)
            .context("lm head forward")?;
        Ok(hidden_states)
    }
}

// ========================================================================= //
//                          Model 
// ========================================================================= //

#[derive(Module)]
pub struct LlamaModel<T: FloatDType> {
    pub embed: Embedding<T>,
    pub layers: Vec<LlamaLayer<T>>,
    pub norm: LlamaRMSNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for LlamaModel<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        let embed_init = init.unwrap_or_else(default_init_linear);
        let embed = Embedding::new(config.vocab_size, config.hidden_size, Some(embed_init))
            .map_err(LlamaError::Nn)
            .context("init embed")?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaLayer::init(config, init)?);
        }
        
        let norm = LlamaRMSNorm::init(config, init).context("init rms norm")?;

        Ok(Self { embed, layers, norm })
    }
}

impl<T: FloatDType> LlamaModel<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let mut hidden_states = self.embed.forward(input_ids)
            .map_err(LlamaError::Core)
            .context("embed forward")?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
            hidden_states = layer.forward(&hidden_states, start_pos, i, cache)
                .with_context(|| format!("layer {i} forward"))?;
        }
        
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        hidden_states = self.norm.forward(&hidden_states).context("norm forward")?;

        Ok(hidden_states)
    }
}

// ========================================================================= //
//                Layer
// ========================================================================= //

#[derive(Module)]
pub struct LlamaLayer<T: FloatDType> {
    pub self_attn: LlamaAttention<T>,
    pub mlp: LlamaMlp<T>,
    pub input_layernorm: LlamaRMSNorm<T>,
    pub post_attention_layernorm: LlamaRMSNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for LlamaLayer<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        let self_attn = LlamaAttention::init(config, init).context("init attention")?;
        let mlp = LlamaMlp::init(config, init).context("init mlp")?;
        let input_layernorm = LlamaRMSNorm::init(config, init).context("init input layernorm")?;
        let post_attention_layernorm = LlamaRMSNorm::new(config).context("init post atten layernorm")?;
        Ok(Self {
            self_attn, mlp, input_layernorm, post_attention_layernorm,
        })
    }
}

impl<T: FloatDType> LlamaLayer<T> {
    /*

                |
                +---------------+
                ^               |
                |               |
    +-----------------------+   |
    |                       |   |
    |           Mlp         |   |
    |                       |   |
    +-----------------------+   |
                ^               |
                |               |
        +---------------+       |
        |    RMSNorm    |       |
        +---------------+       |
                ^               |
                |               |
                +---------------+
                |
                |
                |
                +---------------+
                ^               |
                |               |
    +-----------------------+   |
    |                       |   |
    |        Attention      |   |
    |                       |   |
    +-----------------------+   |
                ^               |
                |               |
        +---------------+       |
        |    RMSNorm    |       |
        +---------------+       |
                ^               |
                |               |
                +---------------+
                |
    
    */
    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut LlamaCache<T>) -> LlamaResult<Tensor<T>> {    
        // Self Attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward(&hidden_states, index_pos, layer_idx, cache).context("self attn forward")?;
        let hidden_states = residual + hidden_states;

        // Mlp
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).context("post attention layer norm forward")?;
        let hidden_states = self.mlp.forward(&hidden_states).context("mlp forward")?;
        let hidden_states = residual + hidden_states;

        Ok(hidden_states)
    }
}

// ========================================================================= //
//                MLP
// ========================================================================= //

#[derive(Module)]
pub struct LlamaMlp<T: FloatDType> {
    pub up_proj: Linear<T>,
    pub gate_proj: Linear<T>,
    pub down_proj: Linear<T>,
}

impl<T: FloatDType> ModuleInit<T> for LlamaMlp<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        let init = init.unwrap_or_else(default_init_linear);

        let up_proj   = Linear::new(config.hidden_size, config.intermediate_size, false, Some(init))?;
        let gate_proj = Linear::new(config.hidden_size, config.intermediate_size, false, Some(init))?;
        let down_proj = Linear::new(config.intermediate_size, config.hidden_size, false, Some(init))?;
    
        Ok(Self { up_proj, down_proj, gate_proj })
    }
}

impl<T: FloatDType> LlamaMlp<T> {
    pub fn forward(&self, x: &Tensor<T>) -> LlamaResult<Tensor<T>> {
        let up = self.up_proj.forward(x)?;
        let gate = self.gate_proj.forward(x)?;
        let up_gate = (up * gate).silu();
        let out = self.down_proj.forward(&up_gate)?;
        Ok(out)
    }
}

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
    pub head_size: usize,
    #[module(skip)]
    pub max_position_embeddings: usize,
}

impl<T: FloatDType> ModuleInit<T> for LlamaAttention<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        // TODO: check 
        let head_size = config.hidden_size / config.num_attention_heads;
        
        let init = init.unwrap_or_else(default_init_linear);
        let q_proj = Linear::new(config.hidden_size, config.hidden_size, false, Some(init))?;
        let k_proj = Linear::new(config.hidden_size, config.num_kv_heads * head_size, false, Some(init))?;
        let v_proj = Linear::new(config.hidden_size, config.num_kv_heads * head_size, false, Some(init))?;
        let o_proj = Linear::new(config.hidden_size, config.hidden_size, false, Some(init))?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_size,
            max_position_embeddings: config.max_position_embeddings,
        })
    }
}

impl<T: FloatDType> LlamaAttention<T> {
    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_attn_heads)
        let k = self.k_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_kv_heads)
        let v = self.v_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_kv_heads)

        // (batch_size, seq_len, head_size * num_attn_heads) => 
        // (batch_size, seq_len, num_attn_heads, head_size) => 
        // (batch_size, num_attn_heads, seq_len, head_size)
        let q = q.reshape((batch_size, seq_len, self.num_attention_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous();

        // (batch_size, seq_len, head_size * num_kv_heads) => 
        // (batch_size, seq_len, num_kv_heads, head_size) => 
        // (batch_size, num_kv_heads, seq_len, head_size)
        let k = k
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous();

        // (batch_size, seq_len, head_size * num_kv_heads) => 
        // (batch_size, seq_len, num_kv_heads, head_size) => 
        // (batch_size, num_kv_heads, seq_len, head_size)
        let mut v = v
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous();

        let q = self.apply_rotary_emb(&q, index_pos, cache)?; // (batch_size, num_attn_heads, seq_len, head_size)
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?; // (batch_size, num_kv_heads, seq_len, head_size)
    
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[layer_idx] {
                // cat [(batch_size, num_kv_heads, cache_seq_len, head_size), (batch_size, num_kv_heads, seq_len, head_size)]
                // => (batch_size, num_kv_heads, total_seq_len, head_size)
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
            cache.kvs[layer_idx] = Some((k.clone(), v.clone()))
        }

        // (batch_size, num_attn_heads, total_seq_len, head_size)
        let (k, v) = self.repeat_kv(&k, &v)?;

        // q: (batch_size, num_attn_heads, seq_len, head_size)
        // k: (batch_size, num_attn_heads, total_seq_len, head_size)
        // v: (batch_size, num_attn_heads, total_seq_len, head_size)

        // (batch_size, num_attn_heads, seq_len, head_size) @ (batch_size, num_attn_heads, total_seq_len, head_size).T 
        // => (batch_size, num_attn_heads, seq_len, total_seq_len)
        // for each token in seq_len, get attn_weight for all total_seq_len
        let attn_weight = q.matmul(&k.transpose_last()?)? / T::from_f64((self.head_size as f64).sqrt());
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

            // (seq_len, total_seq_len) => (batch_size, num_attn_heads, seq_len, total_seq_len)
            let mask = mask.broadcast_as(attn_weight.shape())?;
            mask.if_else(<T as FloatDType>::min_value(), attn_weight)?
        } else {
            attn_weight
        }; // (batch_size, num_attn_heads, seq_len, total_seq_len)

        // (batch_size, num_attn_heads, seq_len, total_seq_len)
        let attn_score = lumen_nn::functional::softmax(&attn_weight_masked, 3)?;

        // (batch_size, num_attn_heads, seq_len, total_seq_len) @ (batch_size, num_attn_heads, total_seq_len, head_size)
        // => (batch_size, num_attn_heads, seq_len, head_size)
        let attn_value = attn_score.matmul(&v)?;
        // (batch_size, num_attn_heads, seq_len, head_size) => 
        // (batch_size, seq_len, num_attn_heads, head_size) => 
        // (batch_size, seq_len, hidden_size)
        let attn_value = attn_value.transpose(1, 2)?.reshape((batch_size, seq_len, hidden_size))?;

        // => (batch_size, seq_len, hidden_size)
        let attn_out = self.o_proj.forward(&attn_value)?;

        Ok(attn_out)
    }

    fn apply_rotary_emb(&self, x: &Tensor<T>, index_pos: usize, cache: &LlamaCache<T>) -> LlamaResult<Tensor<T>> {
        // (batch_size, _n_heads, seq_len, head_size)
        let (_batch_size, _n_heads, seq_len, head_size) = x.dims4()?;

        let cos = cache.cos.narrow(0, index_pos, seq_len)?; // (seq_len, head_size/2)
        let sin = cache.sin.narrow(0, index_pos, seq_len)?; // (seq_len, head_size/2)
        let (cos_seq_len, cos_n_embd) = cos.dims2()?;
        let (sin_seq_len, sin_n_embd) = sin.dims2()?;

        assert_eq!(2 * cos_n_embd, head_size);
        assert_eq!(2 * sin_n_embd, head_size);
        assert!(seq_len <= cos_seq_len);
        assert!(seq_len <= sin_seq_len);

        let cos = Tensor::cat(&[&cos, &cos], 1)?; // (seq_len, head_size)
        let sin = Tensor::cat(&[&sin, &sin], 1)?; // (seq_len, head_size)
        let cos = cos.reshape((1, 1, seq_len, head_size))?; // (1, 1, seq_len, head_size)
        let sin = sin.reshape((1, 1, seq_len, head_size))?; // (1, 1, seq_len, head_size)

        // rotate_half: [-x2, x1]
        let x_rotated = self.rotate_half(x)?;

        // x: (batch_size, _n_heads, seq_len, head_size)
        // x_rotated: (batch_size, _n_heads, seq_len, head_size)
        // cos: (1, 1, seq_len, head_size)
        // sin: (1, 1, seq_len, head_size)
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

// ========================================================================= //
//                RmsNorm
// ========================================================================= //

#[derive(Module)]
pub struct LlamaRMSNorm<T: FloatDType> {
    pub weight: Tensor<T>,
    #[module(skip)]
    pub variance_epsilon: T,
}

impl<T: FloatDType> ModuleInit<T> for LlamaRMSNorm<T> {
    type Config = LlamaConfig;
    type Error = LlamaCtxError;

    fn init(config: &LlamaConfig, init: Option<Init<T>>) -> LlamaResult<Self> {
        let init = init.unwrap_or(Init::ones());
        let weight = init.init((config.hidden_size,))?;
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }
}

impl<T: FloatDType> LlamaRMSNorm<T> {
    pub fn new(config: &LlamaConfig) -> LlamaResult<Self> {
        let weight = Var::ones((config.hidden_size,))?;
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }

    pub fn forward(&self, hidden_states: &Tensor<T>) -> LlamaResult<Tensor<T>> {
        // (xxx, hidden_size) => (xxx, hidden_size)
        let variance = hidden_states.pow(T::two()).mean_keepdim(D::Minus1)?;
        // (xxx, hidden_size) => (xxx, hidden_size) 
        let hidden_states = hidden_states.broadcast_mul(&(variance + self.variance_epsilon))?.sqr();
        // (xxx, hidden_size) => (xxx, hidden_size)
        let out = self.weight.broadcast_mul(&hidden_states)?;
        Ok(out)
    }
}

// ========================================================================= //
//                Default
// ========================================================================= //

#[inline]
fn default_init_linear<T: FloatDType>() -> Init<T> {
    Init::normal(T::zero(), T::from_f64(0.02))
}

// ========================================================================= //
//                Cache
// ========================================================================= //

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
        // => [ 1/rope_theta^{0/head_size}, 1/rope_theta^{2/head_size}, ... ]
        let theta = Tensor::new(theta)?;  // (head_size//2, )
        let theta = theta.reshape((1, theta.element_count()))?; // (1, head_size//2, )

        let idx_theta = Tensor::arange(T::zero(), T::from_usize(config.max_position_embeddings))?;
        let idx_theta = idx_theta.reshape((config.max_position_embeddings, 1))?; // (max_position_embeddings, 1)
        let idx_theta = idx_theta.matmul(&theta)?; // (max_position_embeddings, head_size//2)

        let cos = idx_theta.cos(); // (max_position_embeddings, head_size//2)
        let sin = idx_theta.sin(); // (max_position_embeddings, head_size//2)

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
    let head_size = config.hidden_size / config.num_attention_heads;
    (0..head_size)
        .step_by(2)
        .map(|i| T::one() / T::from_f64(config.rope_theta.powf(i as f64 / head_size as f64)))
        .collect()
}

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use lumen_nn::{init::Init, Module, ModuleInit};
    use crate::llama::{LlamaConfig, LlamaResult};
    use super::{LlamaCache, LlamaForCausalLM};

    #[test]
    fn test_init() {
        let init = Init::<f32>::standard_normal();
        let config = LlamaConfig::test();
        let model = LlamaForCausalLM::init(&config, Some(init)).unwrap();

        for (name, param) in model.named_params() {
            println!("{}: {}", name, param.shape());
        }
    }

    #[test]
    fn test_forward() {
        fn test() -> LlamaResult<()> {
            let config = LlamaConfig::test();
            let model = LlamaForCausalLM::init_default(&config)?;
    
            let input_ids = Tensor::<u32>::new(&[
                [23, 89, 11, 2],
                [90, 12, 43, 29],
            ])?; // (2, 4)
    
            let mut cache = LlamaCache::<f32>::new(true, &config)?;

            println!("{}", input_ids);
            let result = model.forward(input_ids.clone(), 0, &mut cache)?;
            println!("{}", result);
            Ok(())
        }

        if let Err(e) = test() {
            println!("{}", e);
        }
    }
}