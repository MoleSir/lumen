use lumen_core::{FloatDType, IndexOp, IntTensor, Tensor, D};
use lumen_macros::Module;
use lumen_nn::{init::Init, Embedding, Linear, ModuleInit, Parameter};
use thiserrorctx::Context;
use super::{DeepSeekV3Config, DeepSeekV3Error, DeepSeekV3Result};

// ========================================================================= //
//                For Causal LM
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3ForCausalLM<T: FloatDType> {
    pub model: DeepSeekV3Model<T>,
    pub lm_head: Linear<T>, 

    #[module(skip)]
    pub config: DeepSeekV3Config,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3ForCausalLM<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let model = DeepSeekV3Model::init(config, init).context("init llama model")?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false, Some(init.unwrap_or_else(default_init_linear)))
            .map_err(DeepSeekV3Error::Nn)
            .context("init lm head")?;
        
        Ok(Self { model, lm_head, config: config.clone() })
    }
}

// ========================================================================= //
//                          Model 
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3Model<T: FloatDType> {
    pub embed: Embedding<T>,
    pub layers: Vec<DeepSeekV3Layer<T>>,
    pub norm: DeepSeekV3RMSNorm<T>,

    #[module(skip)]
    pub rope: DeepSeekV3Rope<T>,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3Model<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let embed_init = init.unwrap_or_else(default_init_linear);
        let embed = Embedding::new(config.vocab_size, config.hidden_size, Some(embed_init))
            .map_err(DeepSeekV3Error::Nn)
            .context("init embed")?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DeepSeekV3Layer::init(config, init)?);
        }
        
        let norm = DeepSeekV3RMSNorm::init(config, init).context("init rms norm")?;

        let rope = DeepSeekV3Rope::new(config)?;

        Ok(Self { embed, layers, norm, rope })
    }
}

impl<T: FloatDType> DeepSeekV3Model<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, kv_cache: &mut DeepSeekV3KvCache<T>) -> DeepSeekV3Result<Tensor<T>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let mut hidden_states = self.embed.forward(input_ids)
            .map_err(DeepSeekV3Error::Nn)
            .context("embed forward")?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
            hidden_states = layer.forward(&hidden_states, start_pos, i, &self.rope, kv_cache)
                .with_context(|| format!("layer {i} forward"))?;
        }
        
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        hidden_states = self.norm.forward(&hidden_states).context("norm forward")?;

        Ok(hidden_states)
    }
}

// ========================================================================= //
//                Multi Token Perdiction
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3MtpModel<T: FloatDType> {
    pub embed: Embedding<T>,
    pub layers: Vec<DeepSeekV3Layer<T>>,
    pub norm: DeepSeekV3RMSNorm<T>,
    pub mtps: Vec<DeepSeekV3MtpBlock<T>>,

    #[module(skip)]
    pub rope: DeepSeekV3Rope<T>,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3MtpModel<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let embed_init = init.unwrap_or_else(default_init_linear);
        let embed = Embedding::new(config.vocab_size, config.hidden_size, Some(embed_init))
            .map_err(DeepSeekV3Error::Nn)
            .context("init embed")?;
        
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(DeepSeekV3Layer::init(config, init)?);
        }
        
        let norm = DeepSeekV3RMSNorm::init(config, init).context("init rms norm")?;

        let mut mtps = Vec::new();
        for _ in 0..config.num_mtp {
            mtps.push(DeepSeekV3MtpBlock::init(config, init)?);
        }

        let rope = DeepSeekV3Rope::new(config)?;

        Ok(Self { embed, layers, mtps, norm, rope })
    }
}

pub struct DeepSeekV3MtpOutput<T: FloatDType> {
    pub main_output: Tensor<T>,
    pub mtp_outputs: Vec<Tensor<T>>,
}

impl<T: FloatDType> DeepSeekV3MtpModel<T> {
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, kv_cache: &mut DeepSeekV3KvCache<T>) -> DeepSeekV3Result<DeepSeekV3MtpOutput<T>> {
        // embedding: (batch_size, seq_len) -> (batch_size, seq_len, hidden_size)
        let input_states = self.embed.forward(input_ids)
            .map_err(DeepSeekV3Error::Nn)
            .context("embed forward")?;
        let mut hidden_states = input_states.clone();

        // 1. Main
        for (i, layer) in self.layers.iter().enumerate() {
            // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
            hidden_states = layer.forward(&hidden_states, start_pos, i, &self.rope, kv_cache)
                .with_context(|| format!("layer {i} forward"))?;
        }
        
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, hidden_size)
        hidden_states = self.norm.forward(&hidden_states).context("norm forward")?;
        let main_output = hidden_states;

        // Mtps
        let mut pre_hidden_states = main_output.clone();
        let mut mtp_outputs = vec![];
        for mtp in self.mtps.iter() {
            pre_hidden_states = mtp.forward(&pre_hidden_states, &input_states, start_pos, 0, &self.rope, kv_cache)?;
            mtp_outputs.push(pre_hidden_states.clone());
        } 
        
        Ok(DeepSeekV3MtpOutput { main_output, mtp_outputs  })
    }
}

#[derive(Module)]
pub struct DeepSeekV3MtpBlock<T: FloatDType> {
    pub down_proj: Linear<T>,
    pub layer: DeepSeekV3Layer<T>,
    pub rms1: DeepSeekV3RMSNorm<T>,
    pub rms2: DeepSeekV3RMSNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3MtpBlock<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let down_proj = Linear::new(2 * config.hidden_size, config.hidden_size, false, None)?;
        let layer = DeepSeekV3Layer::init(config, init)?;
        let rms1 = DeepSeekV3RMSNorm::new(config)?;
        let rms2 = DeepSeekV3RMSNorm::new(config)?;
        Ok(Self { down_proj, layer, rms1, rms2 })
    }
}

impl<T: FloatDType> DeepSeekV3MtpBlock<T> {
    pub fn forward(&self, pre_hidden_states: &Tensor<T>, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, rope: &DeepSeekV3Rope<T>, kv_cache: &mut DeepSeekV3KvCache<T>) -> DeepSeekV3Result<Tensor<T>> {    
        let pre_hidden_states = self.rms1.forward(pre_hidden_states)?; // (batch, seq_len, hidden_size)
        let hidden_states = self.rms1.forward(hidden_states)?; // (batch, seq_len, hidden_size)
        let hidden_states = Tensor::cat(&[pre_hidden_states, hidden_states], D::Minus1)?; // (batch, seq_len, 2 * hidden_size)
        let hidden_states = self.down_proj.forward(&hidden_states)?; // (batch, seq_len, hidden_size)
        self.layer.forward(&hidden_states, index_pos, layer_idx, rope, kv_cache)
    }
}

// ========================================================================= //
//                Layer
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3Layer<T: FloatDType> {
    pub self_attn: DeepSeekV3Attention<T>,
    pub moe: DeepSeekV3MoE<T>,
    pub input_layernorm: DeepSeekV3RMSNorm<T>,
    pub post_attention_layernorm: DeepSeekV3RMSNorm<T>,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3Layer<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let self_attn = DeepSeekV3Attention::init(config, init).context("init attention")?;
        let moe = DeepSeekV3MoE::init(config, init).context("init moe")?;
        let input_layernorm = DeepSeekV3RMSNorm::init(config, init).context("init input layernorm")?;
        let post_attention_layernorm = DeepSeekV3RMSNorm::init(config, init).context("init post atten layernorm")?;
        Ok(Self {
            self_attn, moe, input_layernorm, post_attention_layernorm,
        })
    }
}

impl<T: FloatDType> DeepSeekV3Layer<T> {
    pub fn forward(&self, hidden_states: &Tensor<T>, index_pos: usize, layer_idx: usize, rope: &DeepSeekV3Rope<T>, kv_cache: &mut DeepSeekV3KvCache<T>) -> DeepSeekV3Result<Tensor<T>> {    
        // Self Attention
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states).context("input layernorm forward")?;
        let hidden_states = self.self_attn.forward(&hidden_states, index_pos, layer_idx, rope, kv_cache).context("self attn forward")?;
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
//                Attention
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3Attention<T: FloatDType> {
    pub q_down_proj: Linear<T>,
    pub q_up_proj: Linear<T>,
    pub q_norm: DeepSeekV3RMSNorm<T>,

    pub kv_down_proj: Linear<T>,
    pub kv_up_proj: Linear<T>,
    pub kv_norm: DeepSeekV3RMSNorm<T>,

    pub o_proj: Linear<T>,

    #[module(skip)]
    pub train: bool,

    #[module(skip)]
    pub config: DeepSeekV3Config,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3Attention<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let init = init.unwrap_or_else(default_init_linear);
        let q_down_proj = Linear::new(config.hidden_size, config.q_lora_size, false, Some(init))?;
        let q_up_proj = Linear::new(config.q_lora_size, config.num_attention_heads * (config.q_rope_size + config.q_nope_size), false, Some(init))?;
        let q_norm = DeepSeekV3RMSNorm::new(config)?;

        let kv_down_proj = Linear::new(config.hidden_size, config.kv_lora_size + config.q_rope_size, false, Some(init))?;
        let kv_up_proj = Linear::new(config.kv_lora_size, config.num_attention_heads * (config.q_nope_size + config.v_size), false, Some(init))?;
        let kv_norm = DeepSeekV3RMSNorm::new(config)?;

        let o_proj = Linear::new(config.num_attention_heads * config.v_size, config.hidden_size, false, Some(init))?;
        
        Ok(Self {
            q_down_proj, q_up_proj, q_norm,
            kv_down_proj, kv_up_proj, kv_norm,
            train: true,
            o_proj,
            config: config.clone(),
        })
    }
}

impl<T: FloatDType> DeepSeekV3Attention<T> {
    pub fn forward(
        &self, 
        hidden_states: &Tensor<T>, 
        index_pos: usize, 
        layer_idx: usize, 
        rope: &DeepSeekV3Rope<T>, 
        kv_cache: &mut DeepSeekV3KvCache<T>
    ) -> DeepSeekV3Result<Tensor<T>> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // 1. 处理 Q
        let q = self.q_down_proj.forward(hidden_states)?; // (batch, seq_len, q_lora_size)
        let q = self.q_norm.forward(&q)?; // (batch, seq_len, q_lora_size)
        let q = self.q_up_proj.forward(&q)?; // (batch, seq_len, n_heads*(q_rope_size+q_nope_size))
        let q = q
            .reshape((batch, seq_len, self.config.num_attention_heads, ()))? // (batch, seq_len, n_heads, (q_rope_size+q_nope_size))
            .transpose(1, 2)?; // (batch, n_heads, seq_len, (q_rope_size+q_nope_size))
        let q_rope = q.narrow(D::Minus1, 0, self.config.q_rope_size)?; // (batch, n_heads, seq_len, q_rope_size)
        let q_nope = q.narrow(D::Minus1, self.config.q_rope_size, self.config.q_nope_size)?; // (batch, n_heads, seq_len, q_nope_size)
        let q_rope = rope.apply_rotary_emb(&q_rope, index_pos)?; // (batch, n_heads, seq_len, q_rope_size)
        let q = Tensor::cat(&[q_nope, q_rope], D::Minus1)?; // (batch, n_heads, seq_len, q_rope_size+q_nope_size)

        // 2. 处理 KV
        let kv = self.kv_down_proj.forward(hidden_states)?; // (batch, seq_len, kv_lora_size+q_rope_size)
        let kv = self.kv_norm.forward(&kv)?; // (batch, seq_len, q_rope_size + kv_lora_size)
        let k_rope = kv.narrow(D::Minus1, 0, self.config.q_rope_size)?.unsqueeze(1)?; // (batch, 1, seq_len, q_rope_size)
        let mut k_rope = rope.apply_rotary_emb(&k_rope, index_pos)?; // (batch, 1, seq_len, q_rope_size)
        let mut kv = kv.narrow(D::Minus1, self.config.q_rope_size, self.config.kv_lora_size)?; // (batch, seq_len, kv_lora_size)
        
        // 3. 保存 kvcache
        if let Some((past_kv, past_k_rope)) = kv_cache.cache[layer_idx].as_ref() {
            kv = Tensor::cat(&[past_kv, &kv], 1)?; // (batch, total_seq_len, kv_lora_size)
            k_rope = Tensor::cat(&[past_k_rope, &k_rope], 2)?; // (batch, 1, total_seq_len, q_rope_size)
        }
        kv_cache.cache[layer_idx] = Some((kv.clone(), k_rope.clone()));
        let total_seq_len = kv.dims()[1]; 

        // 4. 分离 kv
        let k_rope = k_rope.repeat_dim(1, self.config.num_attention_heads)?; // (batch, n_heads, total_seq_len, q_rope_size)
        let kv = self.kv_up_proj.forward(&kv)?; // (batch, total_seq_len, n_heads * (q_nope_size+v_size))
        let kv = kv
            .reshape((batch, seq_len, self.config.num_attention_heads, ()))? // (batch, total_seq_len, n_heads, (q_nope_size+v_size))
            .transpose(1, 2)?; // (batch, n_heads, total_seq_len, q_nope_size+v_size)
        let k_nope = kv.narrow(D::Minus1, 0, self.config.q_nope_size)?; // (batch, n_heads, total_seq_len, q_nope_size)
        let v = kv.narrow(D::Minus1, self.config.q_nope_size, self.config.v_size)?; // (batch, n_heads, total_seq_len, v_size)
        let k = Tensor::cat(&[k_nope, k_rope], D::Minus1)?; // (batch, n_heads, total_seq_len, q_rope_size+q_nope_size)
        
        // 5. 计算 atten
        let d_size = T::from_usize(self.config.q_rope_size + self.config.q_nope_size).sqrt();
        let mut attn_weight = q.matmul(&k.transpose_last()?)? / d_size; // (batch, n_heads, seq_len, total_seq_len)
        if seq_len == total_seq_len {
            /*
                右上角为 true，不包含对角线
                    +--+--+--+
                    |  |xx|xx|
                    +--+--+--+
                    |  |  |xx|
                    +--+--+--+
                    |  |  |  |
                    +--+--+--+
            */
            let mask = Tensor::<bool>::triu(seq_len, false)?; // (seq_len, seq_len)
            attn_weight = mask.if_else(T::MIN_VALUE, &attn_weight)?; // (batch, n_heads, seq_len, seq_len)
        } else {
            // seq_len == total_seq_len => prefill 阶段，之后都是 decode 阶段！每次插入一个新 token，并且也不需要增加 mask
            assert_eq!(seq_len, 1);
        }

        // 6. 计算得分
        let attn_scores = attn_weight.softmax(D::Minus1)?; // (batch, n_heads, seq_len, total_seq_len)
        let context = attn_scores.matmul(&v)?; // (batch, n_heads, seq_len, v_size)
        let context = context
            .transpose(1, 2)? // (batch, seq_len, n_heads, v_size)
            .reshape((batch, seq_len, ()))? // (batch, seq_len, n_heads*v_size)
            .contiguous()?;

        // 7. 计算 output
        let output = self.o_proj.forward(&context)?; // (batch, seq_len, hidden_size)

        Ok(output)
    }
}

// ========================================================================= //
//                MoE
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3MoE<T: FloatDType> {
    pub gate: Linear<T>,
    pub experts: Vec<DeepSeekV3Mlp<T>>,
    pub shared_experts: DeepSeekV3Mlp<T>,

    #[module(skip)]
    pub num_experts_per_tok: usize,
    #[module(skip)]
    pub num_routed_experts: usize,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3MoE<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &Self::Config, init: Option<Init<T>>) -> Result<Self, Self::Error> {
        let gate_init = init.unwrap_or_else(default_init_linear);
        let gate = Linear::new(config.hidden_size, config.num_routed_experts, false, Some(gate_init))
            .map_err(DeepSeekV3Error::Nn)
            .context("init moe gate")?;
    
        let mut experts = Vec::new();
        for i in 0..config.num_routed_experts {
            let expert = DeepSeekV3Mlp::init(
                config.hidden_size, 
                config.intermediate_size, 
                init
            ).with_context(|| format!("init expert {}", i))?;
            experts.push(expert);
        }

        let shared_experts = DeepSeekV3Mlp::init(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            init
        ).context("init shared experts")?;

        Ok(Self {
            gate,
            experts,
            shared_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            num_routed_experts: config.num_routed_experts,
        })
    }
}

impl<T: FloatDType> DeepSeekV3MoE<T> {
    #[allow(unused)]
    pub fn forward(&self, x: &Tensor<T>) -> DeepSeekV3Result<Tensor<T>> {
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        // 1. Shared Experts Path
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, shared_expert_intermediate_size)
        let shared_output = self.shared_experts.forward(x)?;

        // 2.  Router (Gate)
        // (batch_size, seq_len, hidden_size) => (batch_size, seq_len, num_routed_experts)
        let router_logits = self.gate.forward(x)?; 
        // (batch_size, seq_len, num_routed_experts) => (batch_size, seq_len, num_routed_experts)
        // for each token, get `num_routed_experts` prob to each expert
        let routing_probs = lumen_nn::functional::softmax(&router_logits, D::Minus1)?;

        // 3. Top-K Selection
        // (batch_size, seq_len, num_routed_experts) => (batch_size, seq_len, k)
        let (topk_vals, topk_idx) = routing_probs.topk(self.num_experts_per_tok, D::Minus1)?;
        // (batch_size * seq_len, hidden_size)
        let x_flat = x.reshape((batch_size * seq_len, hidden_size))?;
        // (batch_size * seq_len, k)
        let topk_vals = topk_vals.reshape((batch_size * seq_len, self.num_experts_per_tok))?;
        let topk_idx = topk_idx.reshape((batch_size * seq_len, self.num_experts_per_tok))?;

        // (batch_size * seq_len, hidden_size)
        let mut output = x_flat.zeros_like()?;

        // Dispatch to experts
        for i in 0..self.num_experts_per_tok {
            // (batch_size * seq_len, )
            let expert_idx = topk_idx.index((.., i))?; 
            // (batch_size * seq_len, 1)
            let expert_weight = topk_vals.index((.., i))?.unsqueeze(D::Minus1)?; 

            for e in 0..self.num_routed_experts {
                // (batch_size * seq_len, )
                let mask = expert_idx.eq(e as u32)?; 
                if mask.true_count()? == 0 {
                    continue
                }

                // (n_select_token, hidden_size)
                let selected = x_flat.index(&mask)?;
                // (n_select_token, hidden_size)
                let out = self.experts[e].forward(&selected)?;
                // (n_select_token, hidden_size) += (n_select_token, hidden_size)
                output.index(&mask)?.add_(
                    // (n_select_token, 1) * (n_select_token, hidden_size) -> (n_select_token, hidden_size)
                    expert_weight.index(&mask)?.broadcast_mul(&out)?
                )?;
            }

        }

        // (batch_size * seq_len, hidden_size)
        output = output + self.shared_experts.forward(&x_flat)?;
        
        // (batch_size * seq_len, hidden_size) -> ()
        output = output.reshape((batch_size, seq_len, hidden_size))?;

        Ok(output)
    }
}

// ========================================================================= //
//                MLP
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3Mlp<T: FloatDType> {
    pub up_proj: Linear<T>,
    pub gate_proj: Linear<T>,
    pub down_proj: Linear<T>,
}

impl<T: FloatDType> DeepSeekV3Mlp<T> {
    pub fn init(hidden_size: usize, intermediate_size: usize, init: Option<Init<T>>) -> DeepSeekV3Result<Self> {
        let init = init.unwrap_or_else(default_init_linear);
        let up_proj   = Linear::new(hidden_size, intermediate_size, false, Some(init))?;
        let gate_proj = Linear::new(hidden_size, intermediate_size, false, Some(init))?;
        let down_proj = Linear::new(intermediate_size, hidden_size, false, Some(init))?;
    
        Ok(Self { up_proj, down_proj, gate_proj })
    }

    pub fn forward(&self, x: &Tensor<T>) -> DeepSeekV3Result<Tensor<T>> {
        let up = self.up_proj.forward(x)?;
        let gate = self.gate_proj.forward(x)?.silu()?;
        let hidden = up * gate;
        let out = self.down_proj.forward(&hidden)?;
        Ok(out)
    }
}

// ========================================================================= //
//                RmsNorm
// ========================================================================= //

#[derive(Module)]
pub struct DeepSeekV3RMSNorm<T: FloatDType> {
    pub weight: Parameter<T>,
    #[module(skip)]
    pub variance_epsilon: T,
}

impl<T: FloatDType> ModuleInit<T> for DeepSeekV3RMSNorm<T> {
    type Config = DeepSeekV3Config;
    type Error = DeepSeekV3Error;

    fn init(config: &DeepSeekV3Config, init: Option<Init<T>>) -> DeepSeekV3Result<Self> {
        let init = init.unwrap_or(Init::ones());
        let weight = init.init_param((config.hidden_size,))?;
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }
}

impl<T: FloatDType> DeepSeekV3RMSNorm<T> {
    pub fn new(config: &DeepSeekV3Config) -> DeepSeekV3Result<Self> {
        let weight = Parameter::new(Tensor::ones((config.hidden_size,))?);
        let variance_epsilon = T::from_f64(config.rms_norm_eps);
        Ok(Self { weight, variance_epsilon })
    }

    pub fn forward(&self, hidden_states: &Tensor<T>) -> DeepSeekV3Result<Tensor<T>> {
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

pub struct DeepSeekV3Rope<T: FloatDType> {
    pub cos: Tensor<T>,
    pub sin: Tensor<T>,
}

impl<T: FloatDType> DeepSeekV3Rope<T> {
    pub fn new(config: &DeepSeekV3Config) -> DeepSeekV3Result<Self> {
        let theta = calculate_default_inv_freq::<T>(config);
        let theta = Tensor::new(theta)?.reshape((1, config.q_rope_size / 2))?;
        let idx = Tensor::arange(T::zero(), T::from_usize(config.max_position_embeddings))?
            .reshape((config.max_position_embeddings, 1))?;
        let idx_theta = idx.matmul(&theta)?;

        let cos = idx_theta.cos()?; // (max_pos, q_rope_size/2)
        let sin = idx_theta.sin()?; // (max_pos, q_rope_size/2)

        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // (max_pos, q_rope_size)
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?; // (max_pos, q_rope_size)

        Ok(Self { cos, sin, })
    }

    fn apply_rotary_emb(&self, x: &Tensor<T>, index_pos: usize) -> DeepSeekV3Result<Tensor<T>> {
        // (batch_size, _n_heads, seq_len, head_size)
        let (_batch_size, _n_heads, seq_len, head_size) = x.dims4()?;

        let cos = self.cos.narrow(0, index_pos, seq_len)?; // (seq_len, head_size)
        let sin = self.sin.narrow(0, index_pos, seq_len)?; // (seq_len, head_size)

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

    fn rotate_half(&self, x: &Tensor<T>) -> DeepSeekV3Result<Tensor<T>> {
        let last_dim = x.dims().len() - 1;
        let dim_size = x.dims()[last_dim];
        let half_dim = dim_size / 2;

        let x1 = x.narrow(last_dim, 0, half_dim)?;
        let x2 = x.narrow(last_dim, half_dim, half_dim)?;

        let neg_x2 = x2.neg()?;

        let x_rotated = Tensor::cat(&[&neg_x2, &x1], last_dim)?;
        Ok(x_rotated)
    }
}

pub struct DeepSeekV3KvCache<T: FloatDType> {
    pub use_kv_cache: bool,
    pub cache: Vec<Option<(Tensor<T>, Tensor<T>)>>,
}

impl<T: FloatDType> DeepSeekV3KvCache<T> {
    pub fn new(use_kv_cache: bool, config: &DeepSeekV3Config) -> DeepSeekV3Result<Self> {
        Ok(Self { 
            use_kv_cache,
            cache: vec![None; config.num_hidden_layers],
         })
    }
}

fn calculate_default_inv_freq<T: FloatDType>(config: &DeepSeekV3Config) -> Vec<T> {
    (0..config.q_rope_size)
        .step_by(2)
        .map(|i| T::one() / T::from_f64(config.rope_theta.powf(i as f64 / config.q_rope_size as f64)))
        .collect()
}
