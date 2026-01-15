
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
    pub fn gpt2_small() -> Self {
        Self {
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn gpt2_medium() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn gpt2_large() -> Self {
        Self {
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }

    pub fn gpt2_xl() -> Self {
        Self {
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            vocab_size: 50257,
            n_positions: 1024,
            layer_norm_epsilon: 1e-5,
        }
    }
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