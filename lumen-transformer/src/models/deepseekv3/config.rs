#[derive(Clone)]
pub struct DeepSeekV3Config {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,

    // mla
    pub num_attention_heads: usize,
    pub q_lora_size: usize,
    pub q_rope_size: usize,
    pub q_nope_size: usize,
    pub kv_lora_size: usize,
    pub v_size: usize,

    // rope
    pub rms_norm_eps: f64,
    pub rope_theta: f64,

    // moe
    pub num_routed_experts: usize, 
    pub num_experts_per_tok: usize,
    pub num_shared_experts: usize, 
    pub intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    // mtp
    pub num_mtp: usize,
}