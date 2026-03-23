import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置参数
GAMMA = 0.99       # 折扣因子
LAMBDA = 0.95      # GAE 参数
KL_BETA = 0.1      # KL 惩罚系数
BATCH_SIZE = 4
SEQ_LEN = 20       # 假设生成的长度

def compute_gae(rewards, values, gamma=GAMMA, lam=LAMBDA):
    """
    GAE 计算
    rewards: [B, L] - 每个 token 的 reward (包含 KL)
    values:  [B, L] - Critic 对每个 token 的价值预测
    """
    B, L = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gaelam = 0
    
    # 这里的 values 需要包含下一步的预测，通常在序列末尾补 0
    # next_values: [B, L]
    next_values = torch.cat([values[:, 1:], torch.zeros(B, 1).to(values.device)], dim=-1)
    
    # 计算 TD-error: delta = r + gamma * V(s_next) - V(s)
    deltas = rewards + gamma * next_values - values
    
    # 从后往前累加
    for t in reversed(range(L)):
        last_gaelam = deltas[:, t] + gamma * lam * last_gaelam
        advantages[:, t] = last_gaelam
        
    # returns = advantage + value
    returns = advantages + values
    return advantages, returns

def train_token_ppo_step(actor, critic, ref_model, gen_tokens, scores):
    """
    gen_tokens: [B, L] - 包含已生成的 token IDs
    scores:     [B]    - Reward Model 对整句话的打分 (Scalar)
    """
    # 1. 获取 Logits 和 Values
    # logits: [B, L, V], values: [B, L]
    logits = actor(gen_tokens)
    values = critic(gen_tokens)
    
    with torch.no_grad():
        ref_logits = ref_model(gen_tokens)
        old_logits = logits.detach() # 实际应来自 rollout 时的备份

    # 2. 计算 Token-level Log Probs
    # 预测下一个 token，所以 shift 1
    def get_token_logp(lgts, ids):
        log_probs = F.log_softmax(lgts[:, :-1, :], dim=-1) # [B, L-1, V]
        target_ids = ids[:, 1:].unsqueeze(-1)               # [B, L-1, 1]
        return torch.gather(log_probs, 2, target_ids).squeeze(-1)

    actor_logp = get_token_logp(logits, gen_tokens)     # [B, L-1]
    ref_logp   = get_token_logp(ref_logits, gen_tokens) # [B, L-1]
    old_logp   = get_token_logp(old_logits, gen_tokens) # [B, L-1]

    # 3. Reward Shaping (奖励塑造)
    # 计算每个 token 的 KL 散度: kl = log(P/Q)
    # kl_token: [B, L-1]
    kl_token = actor_logp - ref_logp
    
    # 核心公式: reward = -beta * KL
    # 除了最后一位，中间每个 token 的奖励就是 KL 惩罚
    rewards = -KL_BETA * kl_token 
    
    # 将 Reward Model 的总分加在最后一个有效 token 上
    # 注意：scores 是对整句话的评价，我们把它看作是最后一个动作带来的额外奖励
    rewards[:, -1] += scores 

    # 4. 计算 GAE Advantage
    # values 也要对应长度: [B, L-1]
    current_values = values[:, :-1]
    # advantages: [B, L-1], returns: [B, L-1]
    advantages, returns = compute_gae(rewards, current_values)

    # 5. PPO Loss 计算
    # Importance Sampling Ratio
    ratio = torch.exp(actor_logp - old_logp) # [B, L-1]
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value Loss (Critic)
    value_loss = F.mse_loss(current_values, returns.detach())
    
    total_loss = policy_loss + 0.1 * value_loss
    return total_loss