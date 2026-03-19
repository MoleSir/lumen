use lumen_core::IndexOp;
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
        .hidden_size(512)
        .num_hidden_layers(8)
        .build()?;

    let mut model = MiniMindForCausalLM::<f32>::init(&config, None)?;
    let tokenzier = Tokenizer::from_file("./assets/tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let dataset = PretrainDataset::new(
        "./assets/data/pretrain_hq.jsonl", tokenzier, 512
    )?;
    let loader = DataLoader::new(dataset, TensorPairBatcher::default(), BATCH_SIZE, true);

    let mut ad_config = AdamWConfig::<f32>::default();
    ad_config.lr = LR;
    let mut optimizer = AdamW::new(model.params(), ad_config)?;

    let mut cache = MiniMindCache::new(false, &config)?;

    model.train(true);
    for _ in 0..EPOCHS {
        for batch in loader.iter() {
            // (batch_size, seq_len, )
            let (input_ids, label) = batch?;
            let (batch_size, seq_len) = input_ids.dims2()?;

            // (batch_size, seq_len, vocab_size)
            let output_ids = model.forward(input_ids, 0, &mut cache)?;
            // (batch_size * seq_len, vocab_size)
            let output_ids = output_ids.flatten(0, 1)?.index(0..(batch_size * seq_len-1))?; 
            // (batch_size * seq_len, 1) => (batch_size * seq_len - 1, 1)
            let label = label.flatten_all()?.index(1..)?.unsqueeze(1)?;

            let loss = cross_entropy_with_ignore(&output_ids, &label, LossReduction::Sum)?;
            let grads = loss.backward()?;
            optimizer.step(&grads)?;
        }
    }

    Ok(())
}