use std::sync::Arc;

use lumen_core::{FloatDType, IndexOp, IntTensor, NoGradGuard, Tensor, D};
use lumen_dataset::{DataLoader, PairBatcher};
use lumen_nn::{functional::LossReduction, optim::{AdamW, Optimizer}, Linear, Module};
use lumen_transformer::{ForCausalLM, Sampler};
use minimind::{dataset::RLAIDataset, model::{MiniMindCache, MiniMindForCausalLM}, tokenizer::{EncodeOptions, Tokenizer}};

fn main() {
    if let Err(e) = result_main() {
        eprintln!("{e:?}");
    }
}

fn load_sft_model() -> anyhow::Result<(MiniMindForCausalLM<f32>, Arc<Tokenizer>)> {
    todo!()
}

#[derive(Module, Clone)]
pub struct MiniMindForCausalLMWithCritic<T: FloatDType> {
    pub actor_model: MiniMindForCausalLM<T>,
    pub value_head: Linear<T>,
}

impl<T: FloatDType> MiniMindForCausalLMWithCritic<T> {
    pub fn new(actor_model: MiniMindForCausalLM<T>) -> anyhow::Result<Self> {
        let vocab_size = actor_model.config.vocab_size;
        let value_head = Linear::new(vocab_size, 1, false, None)?;
        Ok(Self { actor_model, value_head })
    }
    
    pub fn forward(&self, input_ids: impl Into<IntTensor>, start_pos: usize, cache: &mut MiniMindCache<T>) -> anyhow::Result<(Tensor<T>, Tensor<T>)> {
        // (b, s, hidden) => (b, s, vocab_size)
        let logits = self.actor_model.forward(input_ids, start_pos, cache)?;

        // // (b, s, vocab_size) => (b, s, 1) => (b, s)
        let values = self.value_head.forward(&logits)?.squeeze(D::Minus1)?;
        
        Ok((logits, values))
    }
}

fn result_main() -> anyhow::Result<()> {
    let (actor_model, tokenizer) = load_sft_model()?;
    let model = MiniMindForCausalLMWithCritic::new(actor_model)?;
    let mut old_model = model.actor_model.copy()?;
    let mut ref_model = model.actor_model.copy()?;
    ref_model.eval();
    ref_model.requires_grad(false);

    let dataset = RLAIDataset::new("./assets/cache/xxx", tokenizer.clone())?;
    let loader = DataLoader::new(dataset, PairBatcher::default(), BATCH_SIZE, true);

    const EPOCHS: usize = 1;
    const BATCH_SIZE: usize = 32;
    
    // cache 不进行 kv cache，只保存 rotary embed 的 sin/cos
    let mut optimizer = AdamW::new(model.params(), Default::default())?;
    let mut cache = MiniMindCache::new(false, &model.actor_model.config)?;
    for _ in 0..EPOCHS {
        for batch in loader.iter() {
            let (prompts, _answers) = batch?;
            // prompts: list[string] -> 每个句子都是不完整的，需要 LLM 续写答案
            // 使用 actor_model 对所有问题进行一次生成 (batch, seq_len)
            let gen_out = generate(&model.actor_model, &tokenizer, &prompts)?;

            // 使用标准的 reward 模型对结果打分，每个 batch（句子）一个得分
            let rewards = calculate_rewards(&gen_out, &tokenizer)?; // (batch,)

            // 计算 ref model 对这些完整句子的概率
            let ref_logp = {
                let _gurad = NoGradGuard::new();
                let ref_logits = ref_model.forward(&gen_out, 0, &mut cache)?; // (batch, seq, vocab_size)
                // 同上，计算整个句子的概率
                let softmaxed_ref_logits = lumen_nn::functional::log_softmax(&ref_logits, D::Minus1)?; // (batch, seq)
                let ref_logp = softmaxed_ref_logits.sum(1)?; // (batch,)
                
                ref_logp
            };

            // 计算 old model 对这些完整句子的概率
            let old_logp = {
                let _gurad = NoGradGuard::new();
                let old_logits = old_model.forward(&gen_out, 0, &mut cache)?; // (batch, seq, vocab_size)
                // 同上，计算整个句子的概率
                let softmaxed_old_logp = lumen_nn::functional::log_softmax(&old_logits, D::Minus1)?; // (batch, seq)
                let old_logp = softmaxed_old_logp.sum(1)?; // (batch,)
                
                old_logp
            };

            // On-policy 收集数据，Off-policy 多次更新
            for _ in 0..5 {
                // 虽然我们得到了 actor 模型对每个句子的回答，但每个句子每个 token 的概率我们没有统计，这里需要重新传播（有梯度的）
                // 同时计算每个 batch（句子）的整体概率（完整 tracject 的概率）
                // 这里由于 actor 和 critic 放在一起，所以同时也返回了 critic_model 打分（对生成的 tokens 打分）
                // logits: (batch, seq, vocab_size)
                // values: (batch, seq)
                let (logits, values_seq) = model.forward(&gen_out, 0, &mut cache)?;
                // gen_out 左 shift 1 个位，表示 gen out 预测输出下一个 token id 
                let labels = gen_out.index((.., 1..))?; // (batch, seq_len - 1)
                // 先对每个 token 的 logits 先进行 softmax 转为概率（0-1, 求和为 1），再对每个概率进行 log 操作
                let softmaxed_logits = lumen_nn::functional::log_softmax(&logits, D::Minus1)?;
                // 但这个 token 的预测输出还是有 vocab_size 个，而其实上这个 token 的输出我们已经确定：labels 中的值，我们可以从所有（vocab_size）个概率
                // 提取 label 对应位置 token 的概率，就是在 generate 过程中这个 token 选择了下一个 token 的概率，得到
                // (batch, seq)：表示根据之前 token 预测下一个 token 的概率
                let logp_tokens = softmaxed_logits.gather(labels.unsqueeze(D::Minus1)?, 2)?.squeeze(D::Minus1)?;
                // 将一个 batch（句子）的 log 概率求和，就相当于对原始概率 * 后 log（因为 softmaxed_logits 中的概率值都被 log 过了！）
                let actor_logp = logp_tokens.sum(1)?; // (batch,)

                /*
                    - rewards: 理解为环境给我奖励
                    - values: 利用神经网络预测当前这个状态的价值    
                    二者的差距就是 advantages
                    如果 rewards 超过 values：advantage > 0 说明这个句子超出预期，是好的，我们要增大这个句子生成的概率
                */
                // 只取出 values_seq 的最后一个 token 的 values
                let seq_len = values_seq.dim(2)?;
                let values = values_seq.index((.., seq_len - 1))?; // (batch,)
                let advantages = &rewards - &values; // (batch,)

                // 统计误差：KL + 优势 + reward 和 cir model 的误差

                // 1. KL：actor_logp 表示对 actor_model 每个句子，生成整个回答的总概率的 log，同理对 ref model
                // log 的相减 = / 后 log。也就是用 actor_model 的概率 / ref_model 的概率再 log
                let kl_ref = (&actor_logp - &ref_logp).mean_all()?;
                // actor_model 的概率 / ref_model，重要性采样
                let ratio = (&actor_logp - &old_logp).exp()?; // (batch,)

                // advantages > 0 的句子表示我们要加强这个句子的生成概率，而 ratio 的上面就是 actor_model 生成这个句子的概率。
                // 如果 advantages > 0，backward 就会来提高 actor_model 生成这个句子的概率。
                let surr1 = &advantages * &ratio; // (batch,)
                let surr2 = ratio.clamp(1.-0.1, 1.+0.1)? * &advantages; // 限制增长过大
                let policy_loss = surr1.minimum(surr2)?.neg()?;
                // 训练 crit model
                let value_loss = lumen_nn::functional::mse_loss(&rewards, &values, LossReduction::Mean)?;

                let loss = policy_loss + 0.1 * value_loss + 0.1 * kl_ref;
                let grads = loss.backward()?;

                optimizer.step(&grads)?;
            }

            // 更新 old_model
            old_model = model.actor_model.copy()?;
        }
    }

    Ok(())
}

fn calculate_rewards(gen_out: &Tensor<u32>, tokenizer: &Arc<Tokenizer>) -> anyhow::Result<Tensor<f32>> {
    todo!()
}

fn generate<T: FloatDType>(
    model: &MiniMindForCausalLM<T>, 
    tokenizer: &Arc<Tokenizer>,
    prompts: &[String]
) -> anyhow::Result<Tensor<u32>> 
{
    let _gurad = NoGradGuard::new();
    let sample = Sampler::chat_default();

    let mut result = vec![];
    for prompt in prompts {
        // TODO: 长度对齐、是否返回 input_ids 部分。。。
        let input_ids = tokenizer.encode(&prompt, EncodeOptions::default())?.get_ids().to_vec();
        let output = model.generate(&input_ids, 1024, Some(tokenizer.eos_token_id()), &sample)?;
        result.push(Tensor::new(output)?);
    }

    let enc = Tensor::stack(&result, 0)?;
    Ok(enc)
}