use tokenizers::Tokenizer;


pub fn load_tokenzier() -> anyhow::Result<Tokenizer> {
    Tokenizer::from_file("./data/tokenizer.json").map_err(|e| anyhow::anyhow!(e.to_string()))
}