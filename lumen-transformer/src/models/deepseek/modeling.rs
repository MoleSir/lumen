use std::collections::HashMap;
use lumen_core::{FloatDType, IntTensor, Tensor, Var, D};
use lumen_macros::Module;
use lumen_nn::{init::Initialize, Embedding, Linear};
use thiserrorctx::Context;
use super::{DeepSeekConfig, DeepSeekError, DeepSeekResult};


// ========================================================================= //
//                For Causal LM
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekForCausalLM<T: FloatDType> {
    pub model: DeepSeekModel<T>,
    pub lm_head: Linear<T>, 
}

impl<T: FloatDType> DeepSeekForCausalLM<T> {
    pub fn init(config: &DeepSeekConfig, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let model = DeepSeekModel::init(config, initialize).context("init llama model")?;
        let lm_head = lumen_nn::linear(config.hidden_size, config.vocab_size, false, initialize)
            .map_err(DeepSeekError::Core)
            .context("init lm head")?;
        
        Ok(Self { model, lm_head })
    }

    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut DeepSeekCache<T>) -> DeepSeekResult<Tensor<T>> {
        // (batch_size, seq_len) => (batch_size, seq_len, hidden_size)
        let hidden_states = self.model.forward(input_ids, start_pos, cache).context("model forward")?;
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, vocab_size)
        let hidden_states = self.lm_head.forward(&hidden_states)
            .map_err(DeepSeekError::Core)
            .context("lm head forward")?;
        Ok(hidden_states)
    }
}

// ========================================================================= //
//                          Model 
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekModel<T: FloatDType> {
    pub embed: Embedding<T>,
    pub layers: Vec<DeepSeekLayer<T>>,
    pub norm: DeepSeekRMSNorm<T>,
}

impl<T: FloatDType> DeepSeekModel<T> {
    pub fn init(config: &DeepSeekConfig, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let embed = lumen_nn::embedding(config.vocab_size, config.hidden_size, initialize)
            .map_err(DeepSeekError::Core)
            .context("init embed")?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DeepSeekLayer::init(config, initialize)?);
        }
        
        let norm = DeepSeekRMSNorm::init(config).context("init rms norm")?;

        Ok(Self { embed, layers, norm })
    }

    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut DeepSeekCache<T>) -> DeepSeekResult<Tensor<T>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let mut hidden_states = self.embed.forward(input_ids)
            .map_err(DeepSeekError::Core)
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
pub struct DeepSeekLayer<T: FloatDType> {
    pub self_attn: DeepSeekAttention<T>,
    pub moe: DeepSeekMoE<T>,
    pub input_layernorm: DeepSeekRMSNorm<T>,
    pub post_attention_layernorm: DeepSeekRMSNorm<T>,
}

impl<T: FloatDType> DeepSeekLayer<T> {
    pub fn init(config: &DeepSeekConfig, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let self_attn = DeepSeekAttention::init(config, initialize).context("init attention")?;
        let moe = DeepSeekMoE::init(config, initialize).context("init moe")?;
        let input_layernorm = DeepSeekRMSNorm::init(config).context("init input layernorm")?;
        let post_attention_layernorm = DeepSeekRMSNorm::init(config).context("init post atten layernorm")?;
        Ok(Self {
            self_attn, moe, input_layernorm, post_attention_layernorm,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut DeepSeekCache<T>) -> DeepSeekResult<Tensor<T>> {    
        // Self Attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward(&hidden_states, index_pos, layer_idx, cache).context("self attn forward")?;
        let hidden_states = residual + hidden_states;

        // Mlp
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states).context("post attention layer norm forward")?;
        let hidden_states = self.moe.forward(&hidden_states).context("moe forward")?;
        let hidden_states = residual + hidden_states;

        Ok(hidden_states)
    }
}


// ========================================================================= //
//                MoE
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekMoE<T: FloatDType> {
    pub gate: Linear<T>,
    pub experts: Vec<DeepSeekMlp<T>>,
    pub shared_experts: DeepSeekMlp<T>,

    #[module(skip)]
    pub num_experts_per_tok: usize,
    #[module(skip)]
    pub num_routed_experts: usize,
}

impl<T: FloatDType> DeepSeekMoE<T> {
    pub fn init(config: &DeepSeekConfig, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let gate = lumen_nn::linear(config.hidden_size, config.num_routed_experts, false, initialize)
            .map_err(DeepSeekError::Core)
            .context("init moe gate")?;
        
        let mut experts = Vec::new();
        for i in 0..config.num_routed_experts {
            let expert = DeepSeekMlp::init(
                config.hidden_size, 
                config.intermediate_size, 
                initialize
            ).with_context(|| format!("init expert {}", i))?;
            experts.push(expert);
        }

        let shared_experts = DeepSeekMlp::init(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            initialize
        ).context("init shared experts")?;

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            num_routed_experts: config.num_routed_experts,
        })
    }

    #[allow(unused)]
    pub fn forward(&self, x: &Tensor<T>) -> DeepSeekResult<Tensor<T>> {
        // 1. Shared Experts Path
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, shared_expert_intermediate_size)
        let shared_output = self.shared_experts.forward(x)?;

        // 2.  Router (Gate)
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, num_routed_experts)
        let router_logits = self.gate.forward(x)?; 
        // (batch_size, seq_len, num_routed_experts) => (batch_size, seq_len, num_routed_experts)
        // for each token, get `num_routed_experts` prob to each expert
        let routing_weights = lumen_nn::functional::softmax(&router_logits, D::Minus1)?;

        // 3. Top-K Selection
        unimplemented!()
    }
}

// ========================================================================= //
//                MLP
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekMlp<T: FloatDType> {
    pub up_proj: Linear<T>,
    pub gate_proj: Linear<T>,
    pub down_proj: Linear<T>,
}

impl<T: FloatDType> DeepSeekMlp<T> {
    pub fn init(hidden_size: usize, intermediate_size: usize, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let up_proj   = lumen_nn::linear(hidden_size, intermediate_size, false, initialize)?;
        let gate_proj = lumen_nn::linear(hidden_size, intermediate_size, false, initialize)?;
        let down_proj = lumen_nn::linear(intermediate_size, hidden_size, false, initialize)?;
    
        Ok(Self { up_proj, down_proj, gate_proj })
    }

    pub fn forward(&self, x: &Tensor<T>) -> DeepSeekResult<Tensor<T>> {
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
pub struct DeepSeekAttention<T: FloatDType> {
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

impl<T: FloatDType> DeepSeekAttention<T> {
    pub fn init(config: &DeepSeekConfig, initialize: &Initialize<T>) -> DeepSeekResult<Self> {
        let head_size = config.hidden_size / config.num_attention_heads;
        let q_proj = lumen_nn::linear(config.hidden_size, config.hidden_size, false, initialize)?;
        let k_proj = lumen_nn::linear(config.hidden_size, config.num_kv_heads * head_size, false, initialize)?;
        let v_proj = lumen_nn::linear(config.hidden_size, config.num_kv_heads * head_size, false, initialize)?;
        let o_proj = lumen_nn::linear(config.hidden_size, config.hidden_size, false, initialize)?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_size,
            max_position_embeddings: config.max_position_embeddings,
        })
    }  

    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, cache: &mut DeepSeekCache<T>) -> DeepSeekResult<Tensor<T>> {
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

    fn apply_rotary_emb(&self, x: &Tensor<T>, index_pos: usize, cache: &DeepSeekCache<T>) -> DeepSeekResult<Tensor<T>> {
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

    fn rotate_half(&self, x: &Tensor<T>) -> DeepSeekResult<Tensor<T>> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dims()[last_dim];
        let half_dim = dim_size / 2;

        let x1 = x.narrow(last_dim, 0, half_dim)?;
        let x2 = x.narrow(last_dim, half_dim, half_dim)?;

        let neg_x2 = x2.neg();

        let x_rotated = Tensor::cat(&[&neg_x2, &x1], last_dim)?;
        Ok(x_rotated)
    }

    fn repeat_kv(&self, k: &Tensor<T>, v: &Tensor<T>) -> DeepSeekResult<(Tensor<T>, Tensor<T>)> {
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
pub struct DeepSeekRMSNorm<T: FloatDType> {
    pub weight: Tensor<T>,
    #[module(skip)]
    pub variance_epsilon: T,
}

impl<T: FloatDType> DeepSeekRMSNorm<T> {
    pub fn init(config: &DeepSeekConfig) -> DeepSeekResult<Self> {
        let weight = Var::ones((config.hidden_size,))?;
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }

    pub fn forward(&self, hidden_states: &Tensor<T>) -> DeepSeekResult<Tensor<T>> {
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
//                Cache
// ========================================================================= //

pub struct DeepSeekCache<T: FloatDType> {
    pub use_kv_cache: bool,

    masks: HashMap<usize, Tensor<bool>>,
    kvs: Vec<Option<(Tensor<T>, Tensor<T>)>>,
    cos: Tensor<T>,
    sin: Tensor<T>,
}

impl<T: FloatDType> DeepSeekCache<T> {
    pub fn new(use_kv_cache: bool, config: &DeepSeekConfig) -> DeepSeekResult<Self> {
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

    pub fn mask(&mut self, t: usize) -> DeepSeekResult<Tensor<bool>> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask = Tensor::triu(t, false)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

fn calculate_default_inv_freq<T: FloatDType>(config: &DeepSeekConfig) -> Vec<T> {
    let head_size = config.hidden_size / config.num_attention_heads;
    (0..head_size)
        .step_by(2)
        .map(|i| T::one() / T::from_f64(config.rope_theta.powf(i as f64 / head_size as f64)))
        .collect()
}