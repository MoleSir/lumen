use anyhow::Context;
use lumen_transformer::{gpt2::{Gpt2Config, Gpt2ForCausalLM}, ModuleInit};
use tokenizers::Tokenizer;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("Err: {e:?}");
    }

}

fn result_main() -> anyhow::Result<()> {
    let config = Gpt2Config {
        vocab_size: 50257,
        n_head: 12,
        n_embd: 768,
        n_layer: 12,
        n_positions: 256,
        layer_norm_epsilon: 1e-5,
    };
    
    let mut model = Gpt2ForCausalLM::<f32>::init(&config, None).context("init model")?;

    Ok(())
}