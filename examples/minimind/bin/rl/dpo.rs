use std::sync::Arc;
use lumen_core::{FloatDType, IndexOp, NoGradGuard, Tensor, D};
use lumen_nn::{optim::{AdamW, Optimizer}, Module};
use minimind::{dataset::{DopDataLoader, DpoDataset}, model::{MiniMindCache, MiniMindForCausalLM}, tokenizer::Tokenizer};


fn main() {
    if let Err(e) = result_main() {
        eprintln!("{:?}", e);
    }
}

fn result_main() -> anyhow::Result<()> {
    /*
        1. 加载模型
    */
    let (model, tokenizer) = load_model::<f32>()?;

    // 复制作为 ref model
    let mut ref_model = model.copy()?;
    ref_model.requires_grad(false);
    ref_model.eval();

    /*
        2. 加载数据集
    */
    const BATCH_SIZE: usize = 32;
    let dataset = DpoDataset::new("./assets/cache/xxx", tokenizer.clone(), 512)?;
    let loader = DopDataLoader::from_dataset(dataset, BATCH_SIZE, true);

    /*
        3. 训练循环
    */
    const EPOCHS: usize = 1;
    let mut cache = MiniMindCache::new(false, &model.config)?;
    let mut optimizer = AdamW::new(model.params(), Default::default())?;

    for _ in 0..EPOCHS {
        for batch in loader.iter() {
            let batch = batch?;
            // 拼接 x, y, mask
            let x = Tensor::cat(&[&batch.chosen_x, &batch.rejecten_x], 0)?; // (2*batch, seq)
            let y = Tensor::cat(&[&batch.chosen_y, &batch.rejecten_y], 0)?; // (2*batch, seq)
            let mask = Tensor::cat(&[&batch.chosen_mask, &batch.rejecten_mask], 0)?; // (2*batch, seq)
            
            // 使用 x，ref model 推理
            let ref_log_probs = {
                let _guard = NoGradGuard::new();
                let ref_logits = ref_model.forward(&x, 0, &mut cache)?; // (2*batch, seq, vocab)
                get_per_token_log_probs(&ref_logits, &y)? // (2*batch, seq)
            };

            // 使用 x 进行 model 推理
            let logits = model.forward(&x, 0, &mut cache)?; // (2*batch, seq, vocab)
            let log_probs = get_per_token_log_probs(&logits, &y)?; // (2*batch, seq)
            let loss = dpo_loss(&ref_log_probs, &log_probs, &mask, 0.1)?;

            let grads = loss.backward()?;
            optimizer.step(&grads)?;
        }   
    }

    Ok(())
}

fn load_model<T: FloatDType>() -> anyhow::Result<(MiniMindForCausalLM<T>, Arc<Tokenizer>)> {
    todo!()
}

/// DPO loss 
/// 
/// ## Args
/// - `ref_log_probs`: ref 模型的 token 概率 (2*batch, seq)
/// - `policy_log_probs`: policy 模型的 token 概率 (2*batch, seq)
/// - `mask`: 掩码，只有 asis 输出部分有效 (2*batch, seq)
fn dpo_loss<T: FloatDType>(ref_log_probs: &Tensor<T>, policy_log_probs: &Tensor<T>, mask: &Tensor<bool>, beta: T) -> anyhow::Result<Tensor<T>> {
    // 每个 token 的概率ln -> 通过 mask 遮蔽非 assistance 输出部分
    // 对每个句子的 token 概率 ln 求和 -> 每个句子概率积的 ln（asistant 输出部分）
    let ref_log_probs = mask
        .if_else(ref_log_probs, T::ZERO)?
        .sum(D::Minus1)?; // (2*batch,)
    let policy_log_probs = mask
        .if_else(policy_log_probs, T::ZERO)?
        .sum(D::Minus1)?; // (2*batch,)


    // 将 chosen 和 rejected 数据分开
    let batch = ref_log_probs.dim(0)? / 2;
    let chosen_ref_log_probs = ref_log_probs.narrow(0, 0, batch)?; // (batch, )
    let reject_ref_log_probs = ref_log_probs.narrow(0, batch, batch)?; // (batch,)
    let chosen_policy_log_probs = policy_log_probs.narrow(0, 0, batch)?; // (batch, )
    let reject_policy_log_probs = policy_log_probs.narrow(0, batch, batch)?; // (batch, )

    // 让模型更倾向于 chosen，而不是 rejected, 同时 不要偏离 reference 太远
    // 每个 policy model 对 chosen 的概率，对 reject 的概率
    // 对每个 batch: chosen 句子的概率 ln - reject 句子的概率 ln = ln (chosen 概率 / reject 概率) 
    let pi_logratios = chosen_policy_log_probs - reject_policy_log_probs; // (batch, )
    let ref_logratios = chosen_ref_log_probs - reject_ref_log_probs; // (batch, )
    let logits = pi_logratios - ref_logratios; // (batch, )
    let loss = logsigmoid(&(beta * logits))?.neg()?.mean_all()?;
    Ok(loss)
}

fn logsigmoid<T: FloatDType>(logits: &Tensor<T>) -> anyhow::Result<Tensor<T>> {
    let v = logits.sigmoid()?.ln()?;
    Ok(v)
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
/// - per_token_logps: (batch, seq-1) 每个token概率的 log
fn get_per_token_log_probs<T: FloatDType>(logits: &Tensor<T>, tokens: &Tensor<u32>) -> anyhow::Result<Tensor<T>> {
    let (_, seq, _) = logits.dims3()?;

    // 首先对其 logits 和 tokens（预测的是下一个 token）
    let logits = logits.index((.., ..seq-1, ..))?; // (batch, seq-1, vocab)
    let labels = tokens.index((.., 1..))?.unsqueeze(2)?; // (batch, seq-1, 1)

    // 对 logits -> 转为概率 -> + log
    let logp = lumen_nn::functional::log_softmax(&logits, D::Minus1)?; // (batch, seq-1, vocab)
    
    // 从 logits 中选择出 labels 的概率
    let per_token_logps = logp.gather(&labels, 2)?.squeeze(2)?; // (batch, seq-1)

    Ok(per_token_logps)
}   
