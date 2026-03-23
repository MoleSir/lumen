use tokenizers::{Encoding, TruncationParams};
use super::Tokenizer;

#[derive(Debug, Clone)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub truncation: bool,
    pub max_length: Option<usize>,
    pub padding: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            truncation: false,
            max_length: None,
            padding: false,
        }
    }
}

impl Tokenizer {
    pub fn encode(&self, input: &str, options: EncodeOptions) -> anyhow::Result<Encoding> {
        let mut tokenizer = self.tokenizer.lock().unwrap();

        let save_trunc = if options.truncation {
            let max_length = options.max_length.unwrap_or(
                self.config.extra.get("model_max_length")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2048) as usize
            );

            let save_trunc = tokenizer.get_truncation().cloned();
            tokenizer
                .with_truncation(Some(TruncationParams { max_length, ..Default::default() }))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            
            save_trunc
        } else {
            None
        };

        tokenizer.with_truncation(save_trunc).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    
        let res = tokenizer.encode(input, options.add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;

        Ok(res)
    }
}