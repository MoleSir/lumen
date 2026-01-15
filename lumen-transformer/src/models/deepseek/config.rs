pub struct DeepSeekConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub num_routed_experts: usize, 
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize, 
    pub shared_expert_intermediate_size: usize,
}

impl DeepSeekConfig {
    pub fn deepseek_v2_lite() -> Self {
        Self {
            hidden_size: 2048,
            intermediate_size: 1408, 
            vocab_size: 102400,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_kv_heads: 16,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            
            num_routed_experts: 64,
            num_experts_per_tok: 6,
            num_shared_experts: 2,
            shared_expert_intermediate_size: 2048,
        }
    }

    pub fn deepseek_v2() -> Self {
        Self {
            hidden_size: 5120,
            intermediate_size: 1536, 
            vocab_size: 102400,
            num_hidden_layers: 60,
            num_attention_heads: 128,
            num_kv_heads: 128,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,

            num_routed_experts: 160,
            num_experts_per_tok: 6,
            num_shared_experts: 2,
            shared_expert_intermediate_size: 3072, 
        }
    }

    pub fn deepseek_v3() -> Self {
        Self {
            hidden_size: 7168,
            intermediate_size: 2048,
            vocab_size: 129280, 
            num_hidden_layers: 61,
            num_attention_heads: 128,
            num_kv_heads: 128, 
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,

            num_routed_experts: 256,
            num_experts_per_tok: 8,
            num_shared_experts: 1,
            shared_expert_intermediate_size: 2560, 
        }
    }
}