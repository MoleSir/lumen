
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Qwen2Config {
    pub fn qwen2_0_5b_instruct() -> Self {
        Self {
            hidden_size: 896,
            intermediate_size: 4864,
            vocab_size: 151936,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            tie_word_embeddings: true,
        }
    }
    
    #[allow(unused)]
    pub(crate) fn test() -> Self {
        Self {
            hidden_size: 32,
            intermediate_size: 128,
            vocab_size: 100,
            num_hidden_layers: 8,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_0000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: true,
        }   
    }
}