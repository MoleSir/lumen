import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 假设参数
BATCH_SIZE = 8
SEQ_LEN = 128      # 包含 Prompt + Generated Answer 的总长度
VOCAB_SIZE = 32000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_log_probs(logits, labels):
    """
    计算序列中每个 token 的对数概率并求和得到整句话的 log_p
    logits: [B, S, V]
    labels: [B, S]
    """
    # 1. 预测的是下一个 token，所以将 logits 往右移位，labels 往左移位对齐
    # logits: [B, S-1, V]
    shift_logits = logits[:, :-1, :].contiguous()
    # labels: [B, S-1]
    shift_labels = labels[:, 1:].contiguous()
    
    # 2. 计算 log_softmax: [B, S-1, V]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 3. 根据 labels 索引对应的概率: [B, S-1, 1]
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1))
    
    # 4. 求和得到整句话的对数概率: [B]
    return per_token_logps.squeeze(-1).sum(dim=1)

def train_ppo_step():
    # --- 1. 模型初始化 ---
    # Actor: 策略网络; Critic: 价值网络; Ref: 参考网络 (冻结)
    actor = MyLLM().to(DEVICE)           # [B, S] -> [B, S, V]
    critic = MyCritic().to(DEVICE)       # [B, S] -> [B, S]
    ref_model = MyLLM().to(DEVICE).eval()
    old_actor = MyLLM().to(DEVICE).eval() # 用于 PPO 重要性采样
    
    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-5
    )

    # 模拟输入 Prompts: [B, S_prompt]
    prompts = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 64)).to(DEVICE)

    # --- 2. Rollout 采样阶段 ---
    with torch.no_grad():
        # 生成回答，拼接成完整序列
        # gen_seq: [B, S] (S = prompt_len + gen_len)
        gen_seq = actor.generate(prompts, max_new_tokens=64) 
        
        # 计算 Reward (由外部 Reward Model 打分)
        # rewards: [B]
        rewards = reward_model(gen_seq) 
        
        # 计算 Ref 模型和 Old 模型的概率 (基准)
        # ref_logp: [B], old_logp: [B]
        ref_logits = ref_model(gen_seq)
        ref_logp = get_log_probs(ref_logits, gen_seq)
        
        old_logits = old_actor(gen_seq)
        old_logp = get_log_probs(old_logits, gen_seq)

    # --- 3. PPO 更新阶段 (多次迭代) ---
    for _ in range(5):
        # A. 前向传播
        # current_logits: [B, S, V]
        # current_values_seq: [B, S]
        current_logits = actor(gen_seq)
        current_values_seq = critic(gen_seq)
        
        # B. 计算当前 Actor 的 log_p
        # actor_logp: [B]
        actor_logp = get_log_probs(current_logits, gen_seq)
        
        # C. 计算 Advantage (优势)
        # 取最后一个 token 的 value 作为句子的状态价值估计
        # values: [B]
        values = current_values_seq[:, -1]
        # advantages: [B]
        advantages = rewards - values.detach() 
        # 归一化优点 (可选但推荐)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # D. 计算 Policy Loss (PPO Clip)
        # ratio: [B] (重要性采样比率)
        ratio = torch.exp(actor_logp - old_logp)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        # policy_loss: 标量
        policy_loss = -torch.min(surr1, surr2).mean()

        # E. 计算 Value Loss (Critic)
        # 使 Critic 预测的值接近真实 Reward
        # value_loss: 标量
        value_loss = F.mse_loss(values, rewards)

        # F. 计算 KL Divergence Penalty
        # 惩罚 Actor 偏离 Ref 模型过远
        # kl_div: 标量
        kl_div = (actor_logp - ref_logp).mean()

        # G. 总损失
        total_loss = policy_loss + 0.1 * value_loss + 0.05 * kl_div

        # H. 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Loss: {total_loss.item():.4f} | KL: {kl_div.item():.4f}")

    # 更新 Old Actor 参数
    old_actor.load_state_dict(actor.state_dict())

# --- 辅助模型结构示例 ---
class MyLLM(nn.Module):
    # 标准 Transformer 结构
    def forward(self, x): 
        # x: [B, S] -> logits: [B, S, V]
        pass

class MyCritic(nn.Module):
    # 将 LLM 的 Hidden State 映射到一个标量 Value
    def forward(self, x):
        # x: [B, S] -> values: [B, S]
        pass