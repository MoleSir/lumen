use std::sync::Arc;

use lumen_core::{FloatDType, IndexOp, IntTensor, NoGradGuard, Tensor, D};
use lumen_dataset::{DataLoader, PairBatcher};
use lumen_nn::{functional::LossReduction, optim::{AdamW, Optimizer}, Linear, Module};
use minimind::{dataset::PpoDataset, model::{MiniMindCache, MiniMindForCausalLM}, tokenizer::Tokenizer};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    /*
        1. 加载模型
    */
    // 加载初始模型和 tokenzier
    let (model, tokenizer) = load_model::<f32>()?;

    // 初始化 actor - critor
    let actor_critor = MiniMindForCausalLMWithCritic::new(model.copy()?)?;

    // 初始化 ref / old actor 模型
    let mut old_actor = model.copy()?;
    let mut ref_actor = model;
    // 不需要梯度
    ref_actor.set_train(false);
    ref_actor.requires_grad(false);

    /*
        2. 加载数据集
    */
    const BATCH_SIZE: usize = 32;
    let dataset = PpoDataset::new("./assets/cache/xxx", tokenizer.clone())?;
    let loader = DataLoader::new(dataset, PairBatcher::default(), BATCH_SIZE, true);

    /*
        3. 训练循环
    */
    const EPOCHS: usize = 1;
    const EPSILON: f32 = 0.1;
    let mut cache = MiniMindCache::new(false, &actor_critor.actor.config)?;
    let mut optimizer = AdamW::new(actor_critor.params(), Default::default())?;


    for _ in 0..EPOCHS {
        for batch in loader.iter() {
            let (prompts, answers) = batch?;
            
            // 1. 给每个 prmot 生成一个回答
            let gen_tokens = generate_prompts(&prompts, &actor_critor.actor, &tokenizer)?;

            // 2. 给每个回答一个奖励：这里是每个句子一个奖励！
            let rewrods = calculate_rewards(&gen_tokens, &answers, &tokenizer)?; // (batch,)
        
            // 3. 计算 old/ref 模型的概率
            let ref_logps = {
                let _gurad = NoGradGuard::new();
                let ref_logits = ref_actor.forward(&gen_tokens, 0, &mut cache)?;
                get_log_probs(&ref_logits, &gen_tokens)?
            }; // (batch, )
            let old_logps = {
                let _gurad = NoGradGuard::new();
                let old_logits = old_actor.forward(&gen_tokens, 0, &mut cache)?;
                get_log_probs(&old_logits, &gen_tokens)?
            }; // (batch, )

            // 4. off policy 更新
            for _ in 0..5 {
                // 4.1 使用当前的模型进行推理，获得当前模型的 logits 和 values
                let (logits, values) = actor_critor.forward(&gen_tokens, 0, &mut cache)?;
                let logps = get_log_probs(&logits, &gen_tokens)?;
                let values = values.index((.., D::Minus1))?; // (batch, )

                // 4.2 计算优势函数
                let advantanges = &rewrods - &values.detach(); // (batch, )，没有梯度信息
                
                // 4.3 计算 policy loss
                let ratio = (&logps - &old_logps).exp()?; // (batch, )
                let surr1 = &ratio * &advantanges; // (batch, )
                let surr2 = ratio.clamp(1.0 - EPSILON, 1.0 + EPSILON)? * &advantanges; // (batch, )
                let policy_loss = surr1.minimum(&surr2)?.mean_all()?.neg()?;

                // 4.4 计算 value loss
                let value_loss = lumen_nn::functional::mse_loss(&values, &rewrods, LossReduction::Mean)?;

                // 4.5 KL 散读
                let kl_loss = (&logps - &ref_logps).mean_all()?;

                // 4.5 总 loss
                let loss = policy_loss + 0.1 * value_loss + 0.05 * kl_loss;

                // 4.6 更新 actor critor
                let grads = loss.backward()?;
                optimizer.step(&grads)?;
            }

            // 更新 old model
            old_actor = actor_critor.actor.copy()?;
        }
    }

    Ok(())
}

#[derive(Module)]
pub struct MiniMindForCausalLMWithCritic<T: FloatDType> {
    pub actor: MiniMindForCausalLM<T>,
    pub critor: Linear<T>,
}

impl<T: FloatDType> MiniMindForCausalLMWithCritic<T> {
    pub fn new(actor: MiniMindForCausalLM<T>) -> anyhow::Result<Self> {
        let hidden_size = actor.config.hidden_size;
        let critor = Linear::new(hidden_size, 1, false, None)?;
        Ok(Self { actor, critor })
    }
    
    /// ## Returns
    /// - `logits`: (batch, seq_len, vocab_size)
    /// - `values`: (batch, seq_len,)
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<(Tensor<T>, Tensor<T>)> {
        // hidden: (batch, seq_len, hidden_size)
        let (hidden, logits) = self.actor.forward_with_hidden(input_ids, start_pos, cache)?;
        // (batch, seq_len, hidden_size) => (batch, seq_len, 1) => (batch, seq_len)
        let values = self.critor.forward(&hidden)?.squeeze(D::Minus1)?;
        Ok((logits, values))
    }
}

fn load_model<T: FloatDType>() -> anyhow::Result<(MiniMindForCausalLM<T>, Arc<Tokenizer>)> {
    todo!()
}

/// 给每个 prompt 生成答案序列
/// 
/// ## Returns
/// - `tokens`: (batch, seq)
#[allow(unused)]
fn generate_prompts<T: FloatDType>(prompts: &[String], model: &MiniMindForCausalLM<T>, tokenizer: &Tokenizer) -> anyhow::Result<Tensor<u32>> {
    todo!()
}

#[allow(unused)]
fn calculate_rewards(gen_out: &Tensor<u32>, answers: &[String], tokenizer: &Arc<Tokenizer>) -> anyhow::Result<Tensor<f32>> {
    todo!()
}

/// 计算序列中每个 token 的对数概率并求和得到整句话的 log_p
/// 
/// 给模型输入 `tokens`，模型输出 `logits` 
/// 
/// ## Args
/// - `logits`: (batch, seq, vocab)
/// - `tokens`: (batch, seq)
/// 
/// ## Returns
/// - seq_logps: (batch) 每个句子概率的 log
fn get_log_probs<T: FloatDType>(logits: &Tensor<T>, tokens: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
    let (_, seq, _) = logits.dims3()?;

    // 首先对其 logits 和 tokens（预测的是下一个 token）
    let logits = logits.index((.., ..seq-1, ..))?; // (batch, seq-1, vocab)
    let labels = tokens.index((.., 1..))?.unsqueeze(2)?; // (batch, seq-1, 1)

    // 对 logits -> 转为概率 -> + log
    let logp = lumen_nn::functional::log_softmax(&logits, D::Minus1)?; // (batch, seq-1, vocab)
    
    // 从 logits 中选择出 labels 的概率
    let token_logp = logp.gather(&labels, 2)?.squeeze(2)?; // (batch, seq-1)

    // 每个 token 概率求积后 log -> log 后求和
    let seq_logps = token_logp.sum(1)?; // (batch)

    Ok(seq_logps)
}   