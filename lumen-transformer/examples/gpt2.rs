use std::io::Write;

use futures_util::{pin_mut, StreamExt};
use tokenizers::Tokenizer;
use lumen_transformer::{models::{gpt2::Gpt2ForCausalLM, ForCausalLM}, PretrainedModel, Sampler};

#[tokio::main]
async fn main() {
    if let Err(e) = sync_main() {
        eprintln!("sync error: {e:?}")
    }

    if let Err(e) = async_main().await {
        eprintln!("sync error: {e:?}")
    }
}

fn sync_main() -> anyhow::Result<()> {
    let _gurad = lumen_core::NoGradGuard::new();

    let (tokenizer, model) = load_tokenzier_and_model()?;

    // 准备 Prompt
    let prompt = "Hello, my name is";
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let input_ids = encoding.get_ids();

    // 生成文本 (生成 20 个新 Token)
    println!("Generating...");
    let sampler = Sampler::creative();
    let eos_token = tokenizer.token_to_id("");
    let generated_ids = model.generate(input_ids, 20, eos_token, &sampler)?;

    // 解码并打印结果
    let decoded_text = tokenizer.decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

    println!("\n=== Output ===");
    println!("{}", decoded_text);
    println!("==============\n");

    Ok(())
}


async fn async_main() -> anyhow::Result<()> {
    let _gurad = lumen_core::NoGradGuard::new();

    let (tokenizer, model) = load_tokenzier_and_model()?;

    // 准备 Prompt
    let prompt = "Hello, my name is";
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let input_ids = encoding.get_ids();

    // 生成文本 (生成 20 个新 Token)
    println!("\n=== Output ===");
    let sampler = Sampler::creative();
    let eos_token = tokenizer.token_to_id("");
    let stream = model.generate_stream(input_ids, 20, eos_token, &sampler);
    pin_mut!(stream);
    while let Some(token) = stream.next().await {
        let token = token?;
        let text = tokenizer.decode(&[token], true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
        print!("{}", text);
        std::io::stdout().flush().unwrap();
    }
    println!("\n==============");

    Ok(())
}

fn load_tokenzier_and_model() -> anyhow::Result<(Tokenizer, Gpt2ForCausalLM<f32>)> {
    let tokenizer_path = "./cache/gpt2/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    let model = Gpt2ForCausalLM::<f32>::from_pretrained("./cache/gpt2")?;
    
    println!("Model and Tokenzier loaded successfully!");
    Ok((tokenizer, model))
}