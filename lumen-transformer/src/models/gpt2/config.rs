
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gpt2Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub vocab_size: usize,
    pub n_positions: usize,
    pub layer_norm_epsilon: f64,
}

impl Gpt2Config {
    #[allow(unused)]
    pub(crate) fn test() -> Self {
        Self {
            n_embd: 32,
            n_head: 4,
            n_layer: 8,
            vocab_size: 100,
            n_positions: 64,
            layer_norm_epsilon: 1e-5,
        }
    }
}