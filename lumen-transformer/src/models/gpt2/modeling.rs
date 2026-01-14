use std::collections::HashMap;
use lumen_core::{FloatDType, IntTensor, Tensor, D};
use lumen_macros::Module;
use lumen_nn::{init::Init, Embedding, Linear, ModuleInit, Parameter};
use thiserrorctx::Context;
use super::{Gpt2Config, Gpt2CtxError, Gpt2Error, Gpt2Result};

// ========================================================================= //
//                For Causal LM
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2ForCausalLM<T: FloatDType> {
    pub transformer: Gpt2Model<T>,
    pub lm_head: Linear<T>, 
}

impl<T: FloatDType> ModuleInit<T> for Gpt2ForCausalLM<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let transformer = Gpt2Model::init(config, init).context("init transformer")?;
        let lm_head_init = init.unwrap_or_else(default_init_linear);
        let lm_head = Linear::new(config.n_embd, config.vocab_size, false, Some(lm_head_init))?;
        
        Ok(Self { transformer, lm_head })
    }
}

impl<T: FloatDType> Gpt2ForCausalLM<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut Gpt2Cache<T>) -> Gpt2Result<Tensor<T>> {
        let hidden_states = self.transformer.forward(input_ids, start_pos, cache).context("transformer forward")?;
        let logits = self.lm_head.forward(&hidden_states).map_err(Gpt2Error::Core).context("lm head forward")?;
        Ok(logits)
    }
}

// ========================================================================= //
//                          Model 
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2Model<T: FloatDType> {
    pub wte: Embedding<T>, // Token Embeddings
    pub wpe: Embedding<T>, // Position Embeddings (Learned)
    pub h: Vec<Gpt2Block<T>>,
    pub ln_f: Gpt2LayerNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for Gpt2Model<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let embed_init = init.unwrap_or_else(default_init_linear);
        let wte = Embedding::new(config.vocab_size, config.n_embd, Some(embed_init))?;
        let wpe = Embedding::new(config.n_positions, config.n_embd, Some(embed_init))?;

        let mut h = Vec::new();
        for i in 0..config.n_layer {
            h.push(Gpt2Block::init(config, init).with_context(|| format!("init {i} block"))?);
        }
        
        let ln_f = Gpt2LayerNorm::init(config, init).context("init ln f")?;

        Ok(Self { wte, wpe, h, ln_f })
    }
}

impl<T: FloatDType> Gpt2Model<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut Gpt2Cache<T>) -> Gpt2Result<Tensor<T>> {
        let input_ids: IntTensor = input_ids.into();
        let seq_len = input_ids.dims()[1];

        // Token Embeddings
        // (batch_size, seq_len) => (batch_size, seq_len, n_embd)
        let inputs_embeds = self.wte.forward(input_ids)?;

        // Position Embeddings (Absolute)
        let position_ids = Tensor::arange(start_pos as u32, (start_pos + seq_len) as u32)?; // (seq_len,)
        let position_ids = position_ids.reshape((1, seq_len))?; // (1, seq_len)

        // (1, seq_len) => (1, seq_len, n_embd)
        let position_embeds = self.wpe.forward(position_ids)?; 

        // (batch_size, seq_len, n_embd) + (1, seq_len, n_embd) => (batch_size, seq_len, n_embd) 
        let mut hidden_states = inputs_embeds.broadcast_add(&position_embeds)?;

        // Layers
        for (i, block) in self.h.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, i, cache)?;
        }
        
        // Final Norm
        hidden_states = self.ln_f.forward(&hidden_states)?;

        Ok(hidden_states)
    }
}

// ========================================================================= //
//                Block (Layer)
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2Block<T: FloatDType> {
    pub ln_1: Gpt2LayerNorm<T>,
    pub attn: Gpt2Attention<T>,
    pub ln_2: Gpt2LayerNorm<T>,
    pub mlp: Gpt2MLP<T>,
}

impl<T: FloatDType> ModuleInit<T> for Gpt2Block<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let ln_1 = Gpt2LayerNorm::init(config, init).context("ln 1 init")?;
        let attn = Gpt2Attention::init(config, init).context("attn init")?;
        let ln_2 = Gpt2LayerNorm::init(config, init).context("ln 2 init")?;
        let mlp = Gpt2MLP::init(config, init).context("mlp init")?;

        Ok(Self { ln_1, attn, ln_2, mlp })
    }
}

impl<T: FloatDType> Gpt2Block<T> {
    pub fn forward(&self, hidden_states: &Tensor<T>, layer_idx: usize, cache: &mut Gpt2Cache<T>) -> Gpt2Result<Tensor<T>> {        
        let residual = hidden_states.clone();
        let normalized = self.ln_1.forward(hidden_states).context("ln_1 forward")?;
        let attn_out = self.attn.forward(&normalized, layer_idx, cache).context("attn forward")?;
        let hidden_states = residual + attn_out;

        let residual = hidden_states.clone();
        let normalized = self.ln_2.forward(&hidden_states).context("ln_2 forward")?;
        let mlp_out = self.mlp.forward(&normalized).context("mlp forward")?;
        let hidden_states = residual + mlp_out;

        Ok(hidden_states)
    }
}

// ========================================================================= //
//                MLP
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2MLP<T: FloatDType> {
    pub c_fc: Linear<T>,   // Up projection
    pub c_proj: Linear<T>, // Down projection
}

impl<T: FloatDType> ModuleInit<T> for Gpt2MLP<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let intermediate_size = 4 * config.n_embd;

        let init = init.unwrap_or_else(default_init_linear);
        let c_fc = Linear::new(config.n_embd, intermediate_size, true, Some(init))?;
        let c_proj = Linear::new(intermediate_size, config.n_embd, true, Some(init))?;
    
        Ok(Self { c_fc, c_proj })
    }
}

impl<T: FloatDType> Gpt2MLP<T> {
    pub fn forward(&self, x: &Tensor<T>) -> Gpt2Result<Tensor<T>> {
        let x = self.c_fc.forward(x)?;
        let x = x.gelu(); 
        let x = self.c_proj.forward(&x)?;
        Ok(x)
    }
}

// ========================================================================= //
//                Attention
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2Attention<T: FloatDType> {
    pub c_attn: Linear<T>,
    pub c_proj: Linear<T>,

    #[module(skip)]
    pub n_head: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl<T: FloatDType> ModuleInit<T> for Gpt2Attention<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let head_dim = config.n_embd / config.n_head;

        let c_attn_init = init.unwrap_or_else(default_init_linear);
        let c_attn = Linear::new(config.n_embd, 3 * config.n_embd, true, Some(c_attn_init))?;
        
        let scale = 1.0 / (2.0 * config.n_layer as f64).sqrt();
        let scaled_std = 0.02 * scale;        
        let c_proj_init = Init::normal(T::zero(), T::from_f64(scaled_std));
        let c_proj = Linear::new(config.n_embd, config.n_embd, true, Some(c_proj_init))?;

        Ok(Self {
            c_attn,
            c_proj,
            n_head: config.n_head,
            head_dim,
        })
    }
}

impl<T: FloatDType> Gpt2Attention<T> {
    pub fn forward(&self, hidden_state: &Tensor<T>, layer_idx: usize, cache: &mut Gpt2Cache<T>) -> Gpt2Result<Tensor<T>> {
        let (batch_size, seq_len, n_embd) = hidden_state.dims3()?;

        // (batch_size, seq_len, n_embd) => (batch_size, seq_len, 3 * n_embd)
        let qkv = self.c_attn.forward(hidden_state)?;
        let q = qkv.narrow(2, 0 * n_embd, n_embd)?;
        let k = qkv.narrow(2, 1 * n_embd, n_embd)?;
        let v = qkv.narrow(2, 2 * n_embd, n_embd)?;

        // (batch_size, seq_len, n_embd) => ((batch_size, n_head, seq_len, head_dim)
        let q = self.split_heads(&q, batch_size, seq_len)?;
        let mut k = self.split_heads(&k, batch_size, seq_len)?;
        let mut v = self.split_heads(&v, batch_size, seq_len)?;
    
        // kv cache
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[layer_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?;
                v = Tensor::cat(&[cache_v, &v], 2)?;
            }
            cache.kvs[layer_idx] = Some((k.clone(), v.clone()))
        }

        // attention scores
        // q: (batch, n_head, seq_len, head_dim)
        // k: (batch, n_head, total_seq, head_dim)
        // v: (batch, n_head, total_seq, head_dim)
        let attn_weights = q.matmul(&k.transpose_last()?)?;
        let scale = T::from_f64(1.0 / self.head_dim as f64).sqrt();
        let attn_weights = attn_weights * scale;

        // mask
        let attn_weights_masked = if seq_len > 1 {
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
            let mask = mask.broadcast_as(attn_weights.shape())?;
            mask.if_else(<T as FloatDType>::min_value(), attn_weights)?
        } else {
            attn_weights
        }; // (batch_size, num_attn_heads, seq_len, total_seq)

        // softmax 
        let attn_probs = lumen_nn::functional::softmax(&attn_weights_masked, 3)?;
        // (batch, n_head, seq_len, total_seq) @ (batch, n_head, total_seq, head_dim)
        let attn_output = attn_probs.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()
            .reshape((batch_size, seq_len, self.n_head * self.head_dim))?;

        let attn_output = self.c_proj.forward(&attn_output)?;

        Ok(attn_output)
    }

    fn split_heads(&self, x: &Tensor<T>, batch: usize, seq: usize) -> Gpt2Result<Tensor<T>> {
        let x = x
            .reshape((batch, seq, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous();
        Ok(x)
    }
}

// ========================================================================= //
//                LayerNorm
// ========================================================================= //

#[derive(Module)]
pub struct Gpt2LayerNorm<T: FloatDType> {
    pub weight: Parameter<T>, // Gamma
    pub bias: Parameter<T>,   // Beta
    #[module(skip)]
    pub eps: T,
}

impl<T: FloatDType> ModuleInit<T> for Gpt2LayerNorm<T> {
    type Config = Gpt2Config;
    type Error = Gpt2CtxError;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let size = config.n_embd;
        let eps = T::from_f64(config.layer_norm_epsilon);

        let weight = init
            .unwrap_or(Init::ones())
            .init_param((size, ))
            .map_err(Gpt2Error::Core)
            .context("init weight")?;

        let bias = init
            .unwrap_or(Init::zeros())
            .init_param((size, ))
            .map_err(Gpt2Error::Core)
            .context("init bias")?;

        Ok(Self { weight, bias, eps })
    }
}

impl<T: FloatDType> Gpt2LayerNorm<T> {
    pub fn forward(&self, x: &Tensor<T>) -> Gpt2Result<Tensor<T>> {
        // (batch_size, seq_len, n_dim) => (batch_size, seq_len, 1)
        let mean = x.mean_keepdim(D::Minus1)?;

        // (batch_size, seq_len, n_dim) - (batch_size, seq_len, 1) => (batch_size, seq_len, n_dim)
        let centered = x.broadcast_sub(&mean)?;
        // (batch_size, seq_len, n_dim) => (batch_size, seq_len, 1)
        let variance = centered.pow(T::two()).mean_keepdim(D::Minus1)?;
        
        // norm = (x - mean) / sqrt(var + eps)
        // (batch_size, seq_len, 1)
        let std = (variance + self.eps).sqrt();
        // (batch_size, seq_len, n_dim) / (batch_size, seq_len, 1) => (batch_size, seq_len, n_dim)
        let norm = centered.broadcast_div(&std)?;

        // out = norm * weight + bias
        let out = norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;
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

#[cfg(test)]
mod test {
    use lumen_core::Tensor;
    use lumen_nn::{init::Init, Module, ModuleInit};
    use crate::gpt2::{Gpt2Cache, Gpt2Config, Gpt2Result};

    use super::Gpt2ForCausalLM;

    #[test]
    fn test_init() {
        let init = Init::<f32>::standard_normal();
        let config = Gpt2Config::test();
        let model = Gpt2ForCausalLM::init(&config, Some(init)).unwrap();

        for (name, param) in model.named_params() {
            println!("{}: {}", name, param.shape());
        }

        println!("{}", model);
    }

    #[test]
    fn test_forward() {
        fn test() -> Gpt2Result<()> {
            let config = Gpt2Config::test();
            let model = Gpt2ForCausalLM::init(&config, None)?;
    
            let input_ids = Tensor::<u32>::new(&[
                [23, 89, 11, 2],
                [90, 12, 43, 29],
            ])?; // (2, 4)
    
            let mut cache = Gpt2Cache::<f32>::new(true, &config);

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