mod error;
pub use error::*;
use crate::{components::*, Split, AddedToken, Token};

pub struct Tokenizer<N: Normalize, PT: PreTokenize, M: Model, PP: PostProcess, D: Decode> {
    pub normalizer: N,
    pub pre_tokenizer: PT,
    pub model: M,
    pub post_processor: PP,
    pub decode: D,
    pub added_tokens: Vec<AddedToken>,
}

impl<N, PT, M, PP, D> Tokenizer<N, PT, M, PP, D> 
where 
    M: Model, 
    N: Normalize, 
    PT: PreTokenize, 
    PP: PostProcess,
    D: Decode
{
    pub fn encode(&self, input: &str) -> TokenizeResult<Vec<Token>> {
        let input = self.run_normalize(input)?;
        let splits = self.extract_added_tokens(input)?;
        let splits = self.run_pre_tokenize(splits)?;
        let tokens = self.run_model(splits)?;
        let tokens = self.run_post_process(tokens)?;
        
        Ok(tokens)
    }

    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String, TokenizeError> {
        let mut token_strings = Vec::new();

        for &id in &ids {
            // 1. 优先处理 AddedTokens
            if let Some(at) = self.added_tokens.iter().find(|t| t.id == id) {
                if !skip_special_tokens {
                    token_strings.push(at.content.clone());
                }
            } else {
                // 2. 从 Model 获取原始片段
                if let Some(val) = self.model.id_to_token(id) {
                    token_strings.push(val);
                }
            }
        }

        // 3. 调用 Decoder 组件进行最终清理和合并
        let decoded = self.decode
            .decode(token_strings)
            .map_err(|e| TokenizeError::Decode(Box::new(e)))?;

        Ok(decoded)
    }

    fn run_normalize(&self, input: &str) -> TokenizeResult<String> {
        self.normalizer
            .normalize(input.to_string())
            .map_err(|e| TokenizeError::Normalize(Box::new(e)))
    }

    /// 遍历 input，寻找其中为 AddedToken 的部分，提取转为 Split::AddedToken。并且当场转为 Token
    /// 对其他部分，被 AddedToken 切分，每个子部分封装为 Split::Origin
    fn extract_added_tokens(&self, input: String) -> TokenizeResult<Vec<Split>> {
        let mut splits = vec![];
        let mut cursor = 0;

        while cursor < input.len() {
            let remaining = &input[cursor..];
            let mut best_match: Option<(usize, usize, &AddedToken)> = None;
            for added_token in &self.added_tokens {
                if let Some(local_start) = remaining.find(&added_token.content) {
                    let local_end = local_start + added_token.content.len();
                    
                    match best_match {
                        None => {
                            best_match = Some((local_start, local_end, added_token));
                        }
                        Some((best_start, best_end, _)) => {
                            if local_start < best_start || (local_start == best_start && local_end > best_end) {
                                best_match = Some((local_start, local_end, added_token));
                            }
                        }
                    }
                }
            }

            match best_match {
                Some((local_start, local_end, added_token)) => {
                    let global_match_start = cursor + local_start;
                    let global_match_end = cursor + local_end;
                    splits.push(Split::Origin(input[cursor..global_match_start].to_string()));
                    splits.push(Split::AddedToken(Token { id: added_token.id, value: added_token.content.clone() }));
                    cursor = global_match_end;
                }
                None => {
                    if cursor < input.len() {
                        splits.push(Split::Origin(input[cursor..].to_string()));
                    }
                    break;
                }
            }
        }


        Ok(splits)
    }

    /// 对 parts 中的 Origin 部分，使用 pre_tokenizer 处理，再插入回
    fn run_pre_tokenize(&self, splits: Vec<Split>) -> TokenizeResult<Vec<Split>> {
        let mut new_splits = vec![]; 
        for split in splits.into_iter() {
            match split {
                Split::AddedToken(t) => new_splits.push(Split::AddedToken(t)),
                Split::Origin(s) => {
                    let pts = self.pre_tokenizer
                        .pre_tokenize(s)
                        .map_err(|e| TokenizeError::PreTokenize(Box::new(e)))?;
                    for pt in pts {
                        new_splits.push(Split::Origin(pt));
                    }
                }
            }
        }

        Ok(new_splits)
    }

    fn run_model(&self, splits: Vec<Split>) -> TokenizeResult<Vec<Token>> {
        let mut tokens = vec![]; 
        for split in splits.into_iter() {
            match split {
                Split::AddedToken(t) => tokens.push(t),
                Split::Origin(s) => {
                    let ts = self.model
                        .tokenize(s)
                        .map_err(|e| TokenizeError::Model(Box::new(e)))?;
                    tokens.extend(ts);
                }
            }
        }

        Ok(tokens)
    }

    fn run_post_process(&self, tokens: Vec<Token>) -> TokenizeResult<Vec<Token>> {
        self.post_processor
            .process(tokens)
            .map_err(|e| TokenizeError::PostProcess(Box::new(e)))
    } 
}