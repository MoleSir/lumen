use tokenizers::{EncodeInput, Encoding, PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams, TruncationStrategy};
use super::{Tokenizer, TokenzieResult};

#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    pub padding: Option<PaddingParams>,
    pub truncation: Option<TruncationParams>,
}

impl Tokenizer {
    pub fn encode<'s, E>(&self, inputs: E, add_special_tokens: bool, options: EncodeOptions) -> TokenzieResult<Encoding> 
    where
        E: Into<EncodeInput<'s>>,
    {
        let mut tokenizer = self.tokenizer.lock().unwrap();
    
        // 保存旧状态参数
        let saved_trunc = tokenizer.get_truncation().cloned();
        let saved_padding = tokenizer.get_padding().cloned();

        // 更新输入配置信息
        tokenizer.with_truncation(options.truncation)?;
        tokenizer.with_padding(options.padding);
    
        // encode
        let res = tokenizer.encode(inputs, add_special_tokens)?;
    
        // 恢复状态
        tokenizer.with_truncation(saved_trunc)?;
        tokenizer.with_padding(saved_padding);
    
        Ok(res)
    }
}

impl EncodeOptions {
    pub fn new() -> Self {
        Self::default()
    }

    fn trunc_mut(&mut self) -> &mut TruncationParams {
        self.truncation.get_or_insert_default()
    }

    pub fn truncation(mut self, trunc: TruncationParams) -> Self {
        self.truncation = Some(trunc);
        self
    }

    pub fn max_length(mut self, max_length: usize) -> Self {
        self.trunc_mut().max_length = max_length;
        self
    }

    pub fn truncation_strategy(mut self, strategy: TruncationStrategy) -> Self {
        self.trunc_mut().strategy = strategy;
        self
    }

    pub fn truncation_direction(mut self, direction: TruncationDirection) -> Self {
        self.trunc_mut().direction = direction;
        self
    }

    pub fn stride(mut self, stride: usize) -> Self {
        self.trunc_mut().stride = stride;
        self
    }

    fn padding_mut(&mut self) -> &mut PaddingParams {
        self.padding.get_or_insert_default()
    }

    pub fn padding(mut self, padding: PaddingParams) -> Self {
        self.padding = Some(padding);
        self
    }

    pub fn padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_mut().strategy = strategy;
        self
    }

    pub fn padding_direction(mut self, direction: PaddingDirection) -> Self {
        self.padding_mut().direction = direction;
        self
    }

    pub fn pad_to_multiple_of(mut self, multiple: usize) -> Self {
        self.padding_mut().pad_to_multiple_of = Some(multiple);
        self
    }

    pub fn pad_id(mut self, pad_id: u32) -> Self {
        self.padding_mut().pad_id = pad_id;
        self
    }

    pub fn pad_type_id(mut self, pad_type_id: u32) -> Self {
        self.padding_mut().pad_type_id = pad_type_id;
        self
    }

    pub fn pad_token<S: Into<String>>(mut self, token: S) -> Self {
        self.padding_mut().pad_token = token.into();
        self
    }
}