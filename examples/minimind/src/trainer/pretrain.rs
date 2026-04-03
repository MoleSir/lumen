use lumen_dataset::{DataLoader, TensorPairBatcher};
use lumen_nn::{functional::LossReduction, optim::{AdamW, AdamWConfig, Optimizer}, Module, ModuleInit};
use minimind::{dataset::{cross_entropy_with_ignore, PretrainDataset}, model::{MiniMindCache, MiniMindConfigBuilder, MiniMindForCausalLM}};
use tokenizers::Tokenizer;

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e:?}");
    }
}

fn result_main() -> anyhow::Result<()> {
    const EPOCHS: usize = 1;
    const BATCH_SIZE: usize = 32;
    const LR: f32 = 5e-4;

    let config = MiniMindConfigBuilder::default()
        .hidden_size(128)
        .num_hidden_layers(4)
        .max_position_embeddings(512)
        .build()?;

    let mut model = MiniMindForCausalLM::<f32>::init(&config, None)?;    
    let tokenzier = Tokenizer::from_file("./assets/tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;
    
    let dataset = PretrainDataset::new(
        "./assets/cache/pretrain_hq.jsonl", tokenzier, 512
    )?;
    let loader = DataLoader::new(dataset, TensorPairBatcher::default(), BATCH_SIZE, true);

    let mut ad_config = AdamWConfig::<f32>::default();
    ad_config.lr = LR;
    let mut optimizer = AdamW::new(model.params(), ad_config)?;

    let mut cache = MiniMindCache::new(false, &config)?;

    model.train(true);
    for epoch in 0..EPOCHS {
        println!("--- Epoch {} ---", epoch);
        for (batch_idx, batch) in loader.iter().take(10).enumerate() {
            println!("--- Batch {} ---", batch_idx);

            // (batch_size, seq_len, )
            println!("get batch");
            let (input_ids, label) = batch?;
            println!("{:?}", input_ids.dims());
            println!("{:?}", label.dims());

            // (batch_size, seq_len, vocab_size)
            println!("model forward");
            let start = std::time::Instant::now();
            let output_ids = model.forward(input_ids, 0, &mut cache)?;
            println!("{:?}", std::time::Instant::now() - start);
            
            // (batch_size * seq_len, vocab_size)
            let output_ids = output_ids.flatten(0, 1)?;
            // (batch_size * seq_len, 1)
            let label = label.flatten_all()?.unsqueeze(1)?;
            println!("get loss");
            let loss = cross_entropy_with_ignore(&output_ids, &label, LossReduction::Mean)?;

            println!("backward");
            let start = std::time::Instant::now();
            let grads = loss.backward()?;
            println!("{:?}", std::time::Instant::now() - start);
            
            println!("optimizer step");
            optimizer.step(&grads)?;

            println!(
                "Train Epoch: {} [{}/{} ({:.2}%)]\tLoss: {}",
                epoch, 
                batch_idx * loader.batch_size(), 
                loader.dataset_len(),
                100.0 * batch_idx as f64 / loader.batch_count() as f64, 
                loss.to_scalar()?
            );
        }
    }

    model.save_safetensors("./assets/minimind_test.safetensors")?;

    Ok(())
}