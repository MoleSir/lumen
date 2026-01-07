
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gpt2Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub vocab_size: usize,
    pub n_positions: usize,
    pub layer_norm_epsilon: f64,
}