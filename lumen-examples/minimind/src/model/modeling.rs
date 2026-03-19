use std::{collections::HashMap, str::FromStr};
use anyhow::Context;
use lumen_core::{FloatDType, IntTensor, Tensor, D};
use lumen_nn::{init::Init, Activate, Dropout, Embedding, Linear, Module, ModuleForward, ModuleInit, Parameter};
use super::MiniMindConfig;

// ========================================================================= //
//                For Causal LM
// ========================================================================= //

#[derive(Module)]
pub struct MiniMindForCausalLM<T: FloatDType> {
    pub model: MiniMindModel<T>,
    #[module(skip)]
    pub config: MiniMindConfig,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindForCausalLM<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        let model = MiniMindModel::init(config, init).context("init llama model")?;
        Ok(Self { model, config: config.clone() })
    }
}

impl<T: FloatDType> MiniMindForCausalLM<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<Tensor<T>> {
        // (batch_size, seq_len) => (batch_size, seq_len, hidden_size)
        let hidden_states = self.model.forward(input_ids, start_pos, cache).context("model forward")?;

        let wte_weight = &self.model.embed_tokens.weight;
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, vocab_size)
        let logits = lumen_nn::functional::linear(&hidden_states, &wte_weight, None)
            .context("lm head forward")?;
        
        Ok(logits) 
    }
}

// ========================================================================= //
//                          Model 
// ========================================================================= //

#[derive(Module)]
pub struct MiniMindModel<T: FloatDType> {
    pub embed_tokens: Embedding<T>,
    pub layers: Vec<MiniMindLayer<T>>,
    pub norm: MiniMindRMSNorm<T>,
    pub dropout: Dropout<T>,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindModel<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        let embed_init = init.unwrap_or_else(default_init_linear);
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, Some(embed_init))
            .context("init embed")?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(MiniMindLayer::init(config, init)?);
        }
        
        let norm = MiniMindRMSNorm::init(config, init).context("init rms norm")?;
        let dropout = Dropout::new(T::from_f64(config.dropout));

        Ok(Self { embed_tokens, layers, norm, dropout })
    }
}

impl<T: FloatDType> MiniMindModel<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<Tensor<T>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let hidden_states = self.embed_tokens.forward(input_ids)
            .context("embed forward")?;
        let mut hidden_states = self.dropout.forward(&hidden_states)?;
        
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
pub struct MiniMindLayer<T: FloatDType> {
    pub self_attn: MiniMindAttention<T>,
    pub mlp: MiniMindFeedForward<T>,
    pub input_layernorm: MiniMindRMSNorm<T>,
    pub post_attention_layernorm: MiniMindRMSNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindLayer<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        let self_attn = MiniMindAttention::init(config, init).context("init attention")?;
        let mlp = MiniMindFeedForward::init(config, init).context("init mlp")?;
        let input_layernorm = MiniMindRMSNorm::init(config, init).context("init input layernorm")?;
        let post_attention_layernorm = MiniMindRMSNorm::new(config).context("init post atten layernorm")?;
        Ok(Self {
            self_attn, mlp, input_layernorm, post_attention_layernorm,
        })
    }
}

impl<T: FloatDType> MiniMindLayer<T> {
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
    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<Tensor<T>> {    
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
pub struct MiniMindFeedForward<T: FloatDType> {
    pub up_proj: Linear<T>,
    pub gate_proj: Linear<T>,
    pub down_proj: Linear<T>,
    pub dropout: Dropout<T>,
    pub act_fn: Activate,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindFeedForward<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        let init = init.unwrap_or_else(default_init_linear);

        let intermediate_size = config.intermediate_size();
        let up_proj   = Linear::new(config.hidden_size, intermediate_size, false, Some(init))?;
        let gate_proj = Linear::new(config.hidden_size, intermediate_size, false, Some(init))?;
        let down_proj = Linear::new(intermediate_size, config.hidden_size, false, Some(init))?;
        let dropout = Dropout::new(T::from_f64(config.dropout));
        let act_fn = Activate::from_str(&config.hidden_act)?;
        
        Ok(Self { up_proj, down_proj, gate_proj, dropout, act_fn })
    }
}

impl<T: FloatDType> MiniMindFeedForward<T> {
    pub fn forward(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let up = self.up_proj.forward(x)?;
        let gate = self.act_fn.forward(self.gate_proj.forward(x)?)?;
        let hidden = up * gate;
        let out = self.down_proj.forward(&hidden)?;
        let dropout_out = self.dropout.forward(&out)?;
        Ok(dropout_out)
    }
}

// ========================================================================= //
//                Attention
// ========================================================================= //

#[derive(Module)]
pub struct MiniMindAttention<T: FloatDType> {
    pub q_proj: Linear<T>,
    pub k_proj: Linear<T>,
    pub v_proj: Linear<T>,
    pub o_proj: Linear<T>,
    pub attn_dropout: Dropout<T>,
    pub resid_dropout: Dropout<T>,

    #[module(skip)]
    pub num_attention_heads: usize,
    #[module(skip)]
    pub num_key_value_heads: usize,
    #[module(skip)]
    pub head_size: usize,
    #[module(skip)]
    pub max_position_embeddings: usize,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindAttention<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        // TODO: check 
        let head_size = config.hidden_size / config.num_attention_heads;
        
        let init = init.unwrap_or_else(default_init_linear);
        let q_proj = Linear::new(config.hidden_size, config.hidden_size, false, Some(init))?;
        let k_proj = Linear::new(config.hidden_size, config.num_key_value_heads * head_size, false, Some(init))?;
        let v_proj = Linear::new(config.hidden_size, config.num_key_value_heads * head_size, false, Some(init))?;
        let o_proj = Linear::new(config.hidden_size, config.hidden_size, false, Some(init))?;
        let attn_dropout = Dropout::new(T::from_f64(config.dropout));
        let resid_dropout = Dropout::new(T::from_f64(config.dropout));

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            attn_dropout, resid_dropout,

            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_size,
            max_position_embeddings: config.max_position_embeddings,
        })
    }
}

impl<T: FloatDType> MiniMindAttention<T> {
    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<Tensor<T>> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_attn_heads)
        let k = self.k_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_key_value_heads)
        let v = self.v_proj.forward(hidden_states)?; // (batch_size, seq_len, head_size * num_key_value_heads)

        // (batch_size, seq_len, head_size * num_attn_heads) => 
        // (batch_size, seq_len, num_attn_heads, head_size) => 
        // (batch_size, num_attn_heads, seq_len, head_size)
        let q = q.reshape((batch_size, seq_len, self.num_attention_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;

        // (batch_size, seq_len, head_size * num_key_value_heads) => 
        // (batch_size, seq_len, num_key_value_heads, head_size) => 
        // (batch_size, num_key_value_heads, seq_len, head_size)
        let k = k
            .reshape((batch_size, seq_len, self.num_key_value_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;

        // (batch_size, seq_len, head_size * num_key_value_heads) => 
        // (batch_size, seq_len, num_key_value_heads, head_size) => 
        // (batch_size, num_key_value_heads, seq_len, head_size)
        let mut v = v
            .reshape((batch_size, seq_len, self.num_key_value_heads, self.head_size))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?; // (batch_size, num_attn_heads, seq_len, head_size)
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?; // (batch_size, num_key_value_heads, seq_len, head_size)
    
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[layer_idx] {
                // cat [(batch_size, num_key_value_heads, cache_seq_len, head_size), (batch_size, num_key_value_heads, seq_len, head_size)]
                // => (batch_size, num_key_value_heads, total_seq_len, head_size)
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                
                // if too long
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(2, k_seq_len - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous()?;
                }

                let v_seq_len = k.dims()[2];
                if v_seq_len > self.max_position_embeddings {
                    v = v
                        .narrow(2, v_seq_len - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous()?;
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
            mask.if_else(T::MIN_VALUE, attn_weight)?
        } else {
            attn_weight
        }; // (batch_size, num_attn_heads, seq_len, total_seq_len)

        // (batch_size, num_attn_heads, seq_len, total_seq_len)
        let attn_score = lumen_nn::functional::softmax(&attn_weight_masked, 3)?;
        let attn_score = self.attn_dropout.forward(&attn_score)?;

        // (batch_size, num_attn_heads, seq_len, total_seq_len) @ (batch_size, num_attn_heads, total_seq_len, head_size)
        // => (batch_size, num_attn_heads, seq_len, head_size)
        let attn_value = attn_score.matmul(&v)?;
        // (batch_size, num_attn_heads, seq_len, head_size) => 
        // (batch_size, seq_len, num_attn_heads, head_size) => 
        // (batch_size, seq_len, hidden_size)
        let attn_value = attn_value.transpose(1, 2)?.reshape((batch_size, seq_len, hidden_size))?;

        // => (batch_size, seq_len, hidden_size)
        let attn_out = self.o_proj.forward(&attn_value)?;
        let attn_out = self.resid_dropout.forward(&attn_out)?;

        Ok(attn_out)
    }

    fn apply_rotary_emb(&self, x: &Tensor<T>, index_pos: usize, cache: &MiniMindCache<T>) -> anyhow::Result<Tensor<T>> {
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

    fn rotate_half(&self, x: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dims()[last_dim];
        let half_dim = dim_size / 2;

        let x1 = x.narrow(last_dim, 0, half_dim)?;
        let x2 = x.narrow(last_dim, half_dim, half_dim)?;

        let neg_x2 = x2.neg()?;

        let x_rotated = Tensor::cat(&[&neg_x2, &x1], last_dim)?;
        Ok(x_rotated)
    }

    fn repeat_kv(&self, k: &Tensor<T>, v: &Tensor<T>) -> anyhow::Result<(Tensor<T>, Tensor<T>)> {
        let k = self.repeat(k)?;
        let v = self.repeat(v)?;
        Ok((k, v))
    }

    fn repeat(&self, k: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        let repeat_times = self.num_attention_heads / self.num_key_value_heads;
        let (batch_size, _num_key_value_heads, seq_len, head_size) = k.dims4()?;
        let k = k.unsqueeze(2)?; 
        let k = k.repeat_dim(2, repeat_times)?;
        let k = k.reshape((batch_size, self.num_attention_heads, seq_len, head_size))?;
        Ok(k)
    }
}

// ========================================================================= //
//                RmsNorm
// ========================================================================= //

#[derive(Module)]
pub struct MiniMindRMSNorm<T: FloatDType> {
    pub weight: Parameter<T>,
    #[module(skip)]
    pub variance_epsilon: T,
}

impl<T: FloatDType> ModuleInit<T> for MiniMindRMSNorm<T> {
    type Config = MiniMindConfig;
    type Error = anyhow::Error;

    fn init(config: &MiniMindConfig, init: Option<Init<T>>) -> anyhow::Result<Self> {
        let init = init.unwrap_or(Init::ones());
        let weight = init.init_param((config.hidden_size,))?;
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }
}

impl<T: FloatDType> MiniMindRMSNorm<T> {
    pub fn new(config: &MiniMindConfig) -> anyhow::Result<Self> {
        let weight = Parameter::new(Tensor::ones((config.hidden_size,))?);
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }

    pub fn forward(&self, hidden_states: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
        // (xxx, hidden_size) => (xxx, hidden_size)
        let variance = hidden_states.pow(T::two())?.mean_keepdim(D::Minus1)?;
        // (xxx, hidden_size) => (xxx, hidden_size) 
        let rms = (variance + self.variance_epsilon).sqrt()?;
        let hidden_states = hidden_states.broadcast_div(&rms)?;
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

pub struct MiniMindCache<T: FloatDType> {
    pub use_kv_cache: bool,

    masks: HashMap<usize, Tensor<bool>>,
    kvs: Vec<Option<(Tensor<T>, Tensor<T>)>>,
    cos: Tensor<T>,
    sin: Tensor<T>,
}

impl<T: FloatDType> MiniMindCache<T> {
    pub fn new(use_kv_cache: bool, config: &MiniMindConfig) -> anyhow::Result<Self> {
        let theta = calculate_default_inv_freq::<T>(config);
        
        //    [0, 2, 4, ...] 
        // => [ 1/rope_theta^{0/head_size}, 1/rope_theta^{2/head_size}, ... ]
        let theta = Tensor::new(theta)?;  // (head_size//2, )
        let theta = theta.reshape((1, theta.element_count()))?; // (1, head_size//2, )

        let idx_theta = Tensor::arange(T::zero(), T::from_usize(config.max_position_embeddings))?;
        let idx_theta = idx_theta.reshape((config.max_position_embeddings, 1))?; // (max_position_embeddings, 1)
        let idx_theta = idx_theta.matmul(&theta)?; // (max_position_embeddings, head_size//2)

        let cos = idx_theta.cos()?; // (max_position_embeddings, head_size//2)
        let sin = idx_theta.sin()?; // (max_position_embeddings, head_size//2)

        Ok(Self { 
            use_kv_cache,
            masks: HashMap::new(),
            kvs: vec![None; config.num_hidden_layers],
            cos, sin,
         })
    }

    pub fn mask(&mut self, t: usize) -> anyhow::Result<Tensor<bool>> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask = Tensor::triu(t, false)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

fn calculate_default_inv_freq<T: FloatDType>(config: &MiniMindConfig) -> Vec<T> {
    let head_size = config.hidden_size / config.num_attention_heads;
    (0..head_size)
        .step_by(2)
        .map(|i| T::one() / T::from_f64(config.rope_theta.powf(i as f64 / head_size as f64)))
        .collect()
}

// #[cfg(test)]
// mod test {
//     use lumen_core::Tensor;
//     use lumen_nn::{init::Init, Module, ModuleInit};
//     use crate::{llama::{MiniMindConfig, anyhow::Result}, ForCausalLM};
//     use super::{MiniMindCache, MiniMindForCausalLM};

//     #[test]
//     fn test_init() {
//         let init = Init::<f32>::standard_normal();
//         let config = MiniMindConfig::test();
//         let model = MiniMindForCausalLM::init(&config, Some(init)).unwrap();

//         for (name, param) in model.named_params() {
//             println!("{}: {}", name, param.shape());
//         }
//     }

//     #[test]
//     fn test_forward() {
//         fn test() -> anyhow::Result<()> {
//             let config = MiniMindConfig::test();
//             let model = MiniMindForCausalLM::init_default(&config)?;
    
//             let input_ids = Tensor::<u32>::new(&[
//                 [23, 89, 11, 2],
//                 [90, 12, 43, 29],
//             ])?; // (2, 4)
    
//             let mut cache = MiniMindCache::<f32>::new(true, &config)?;

//             println!("{}", input_ids);
//             let result = model.forward(input_ids.clone(), 0, &mut cache)?;
//             println!("{}", result);
//             Ok(())
//         }

//         if let Err(e) = test() {
//             println!("{}", e);
//         }
//     }
// }