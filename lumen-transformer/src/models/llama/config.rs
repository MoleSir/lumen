
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl LlamaConfig {
    pub fn llama_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10_0000.0,
            max_position_embeddings: 4096,
        }
    }

    pub fn llama2_7b() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_0000.0,
            max_position_embeddings: 4096,
        }
    }

    pub(crate) fn test() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 128,
            vocab_size: 32000,
            num_hidden_layers: 8,
            num_attention_heads: 4,
            num_kv_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_0000.0,
            max_position_embeddings: 64,
        }   
    }
}