# PPO 算法

## 模型期望

每个结果可能的概率和结果值乘积之和：
$$
E(x)_{x \sim p(x) } = \sum_x x \cdot p(x)
$$
可以采样多次取平均值进行近似：
$$
E(x) _{x \sim p(x)} \approx \frac 1 N \sum_{i=1}^N x \sim p(x)
$$
对强化学习模型，目标是让 Agent 执行一次完整游戏后的 Return 最大，写为期望的形式：
$$
E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \sum _{\tau} R(\tau) P_{\theta }(\tau)\tag{1}
$$

- $\tau$ 表示整个游戏的完整路径，包含若干个动作
- $R(\tau)$ 表示环境对当前这个完整路径给出的 Return
- $P_{\theta}(\tau)$ 表示在当前模型参数 $\theta$ 下，出现这个完整路径的概率

也比较好理解，就是奖励的期望是奖励的值 * 路径的概率，再求和



## 期望的导数

目标是找到参数 $\theta$，使得 $(1)$ 最大，利用梯度上升，计算 $(1)$ 对参数的导数：
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \nabla \sum _{\tau} R(\tau) P_{\theta }(\tau)
$$
假设环境给某个路径的奖励 $R(\tau)$ 和模型本身没有关系：
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \sum _{\tau} R(\tau) \nabla P_{\theta }(\tau)
\\\\= \sum _{\tau} R(\tau) \nabla P_{\theta }(\tau) \frac{P_{\theta }(\tau)}{P_{\theta }(\tau)}
\\\\= \sum _{\tau} \left ( R(\tau)\frac{ \nabla P_{\theta }(\tau)}{P_{\theta }(\tau)}  \right ) P_{\theta }(\tau)
$$
这个式子最后成为：某个值 * 概率的形式，等价为一个新的数学期望，那么根据蒙特卡洛，我们可以让这个值进行多次采样，然后取平均
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \frac 1 N \sum_{i=1}^N R(\tau)\frac{ \nabla P_{\theta }(\tau)}{P_{\theta }(\tau)}
$$
再利用 $log$ 函数求导法则：
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \frac 1 N \sum_{n=1}^N R(\tau_n) \nabla \ln P_{\theta}(\tau_n)\tag{2}
$$
其中 $\tau_n \sim p_{\theta }(\tau)$，表示第 $n$ 个路径。

进一步，这里的 $P_{\theta}(\tau_n)$ 是由多个动作的概率组成，假设每个动作彼此无关，那么可以写为每个动作连乘形式：
$$
P_{\theta}(\tau_n) = \prod _{t=1}^T P_{\theta}(a_{t}^n \mid s_{t}^n)
$$
其中 $P_{\theta}(a_{t}^n \mid s_{t}^n)$ 表示在状态 $s_t$ 的情况下，根据当前参数，选择动作 $a_t$ 的概率，带入 $(2)$：
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \frac 1 N \sum_{n=1}^N R(\tau_n) \nabla \ln P_{\theta}(\tau_n)
\\\\ = \frac 1 N \sum_{n=1}^N R(\tau_n) \nabla \ln \prod _{t=1}^T P_{\theta}(a_{t}^n \mid s_{t}^n)
\\\\ = \frac 1 N \sum_{n=1}^N R(\tau_n) \sum _{t=1}^T \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)
\\\\ = \frac 1 N \sum_{n=1}^N  \sum _{t=1}^T R(\tau_n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n) \tag{3}
$$
根据梯度上升，为了增大一个函数，当某个位置函数导数为正，我们应该继续增大输入。对 $(3)$，如果某个 $\tau_n$ 的回报 $R(\tau_n) > 0$，而 $\ln$ 函数本身是单调上升的，所以 $\nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n) $ 部分也 $>0$，此时导数大于 0，为了使得函数增大，应该使得彼 $\theta$ 向着 $P_{\theta}(a_{t}^n \mid s_{t}^n)$  增大而变化。

直观理解：此时回报大于 0，我们应该增大这些动作的概率！



## 状态/动作价值

根据 $(3)$：
$$
\nabla E(R(\tau)) _{\tau \sim p_{\theta}(\tau)} = \frac 1 N \sum_{n=1}^N  \sum _{t=1}^T R(\tau_n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)
$$
当某个 $\tau_n$ 的回报比较大，就会增大这个路径上所有动作的概率！这不合理，虽然整个路径回报是正，但可能其中某些动作是负收益，被其他更强的正收益拉回了。直接使用 $R(\tau_n)$ 使得路径上不好的动作也被提高了概率。

我们应该将这个替换为每个动作自己的 Return。定义某个 $t$ 时候的 Return：
$$
R_{t}^n = \sum_{t'=t}^T \gamma ^{t' - t} r^n_{t'}
$$
其中 $\gamma $ 表示衰减因子，$r_{t'}$ 表示 $t'$ 时刻执行动作得到的立即奖励。意思就是将当前时刻的奖励加之后所有的奖励，但之后的奖励需要每间隔一次时刻加上一个衰减因子，这样使得 $t'$ 时刻即包含了之后奖励信息，但又主要关注其附件的奖励。更新 $(3)$
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T R_t^n \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{4}
$$
现在对每个动作都有独立的 Return。

进一步，我们本意是要考察某个动作的好坏，但如果直接使用环境给的奖励，还是有可能出问题：比如现在 Agent 在一个好的局势，不管我做什么动作奖励都是正的（包括本身是不好的动作）；反之如果当前局势很坏，有的动作可以逆转局势，但得到的即时奖励还是负的。

我们需要一个优势的概率：将当前动作的 $R_{t}^n$ 减去一个 baseline，baseline 表示你在当前状态下执行任意动作的平均奖励，这样才可以评估不同动作的好坏之分：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T (R_t^n - B(s_n^t)) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{5}
$$
 其中 $B(s_n^t)$ 表示在状态 $s_n^t$ 下平均的价值。

综上，这了引入两个概念：

- Action-Value Function：动作价值函数 $Q_{\theta}(s, a)$，在某个状态 s 下，做出动作 a，期望的 Return。
- State-Value Function：状态价值函数 $V_{\theta}(s)$，某个状态期望的回报。

定义优势函数：
$$
A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)\tag{6}
$$
也就是说在 s 状态下，我真的采取一个动作得到的回报 $Q_{\theta}(s, a)$，可以比我在这个状态下平均得到的回报 $V_{\theta}(s)$ 超出多少。这样可以衡量在状态 s 下，采取 a 动作相比其他动作的优势。

最后使用优势函数给出：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A_{\theta} (s_{t}^n, a_{t}^n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{7}
$$
即：如果在 s 下执行 a 的优势大，我们就增大这个概率。



## GAE

对动作价值函数 $Q_{\theta}(s, a)$ 可以做以下近似：
$$
Q_{\theta} (s, a) = r_t + \gamma \cdot V_{\theta}(s_{t+1})
$$
理解：在 s 状态执行 a，得到 r 回报，并且进入状态 $s_{t+1}$，所以之后所有的 Return 都使用状态价值函数 $V_{\theta}(s_{t+1})$ 代替。带入优势函数公式 $(6)$：
$$
A_{\theta}(s, a) = Q_{\theta}(s, a) - V_{\theta}(s)
 =  r_t + \gamma \cdot V_{\theta}(s_{t+1}) - V_{\theta}(s)
$$
那么同理，除了直接用 $V_{\theta}(s_{t+1})$ 来近似之后所有的回报，我们可以多 action 几次：
$$
Q_{\theta}(s_t, a) = r_t + \gamma r_{t+1} + \gamma^2V_{\theta }(s_{t+2})
$$
也可以带入 $(6)$：
$$
A_{\theta}(s_t, a) =  r_t + \gamma r_{t+1} + \gamma^2V_{\theta }(s_{t+2}) -V_{\theta}(s_t)
$$
这个采样可以一直进行下去（有点类似蒙特卡洛和时序差分的混合）：
$$
Q_{\theta}(s_t, a) = r_t + \gamma t_{t+1} + \cdots + \gamma ^{T-1} r_{T-1} + \gamma ^T V_{\theta}(s_{t+T})
$$

$$
A_{\theta}(s_t, a) = r_t + \gamma t_{t+1} + \cdots + \gamma ^{T-1} r_{T-1} + \gamma ^T V_{\theta}(s_{t+T}) - V_{\theta}(s_t)
$$

可以写为一系列式子：
$$
A^1_{\theta}(s_t, a) =r_t + \gamma \cdot V_{\theta}(s_{t+1}) - V_{\theta}(s) \\\\

A^2_{\theta}(s_t, a) =  r_t + \gamma r_{t+1} + \gamma^2V_{\theta }(s_{t+2}) -V_{\theta}(s) \\\\

\cdots \\\\

A^T_{\theta}(s_t, a) = r_t + \gamma t_{t+1} + \cdots + \gamma ^{T-1} r_{t+T-1} + \gamma ^T V_{\theta}(s_{t+T}) - V_{\theta}(s)\tag{8}
$$
化简这个式子，定义一个中间变量：
$$
\delta _{t} = r_t + \gamma V_{\theta }(s_{t+1}) - V_{\theta}(s_t)\\\\
\delta _{t+1} = r_{t+1} + \gamma V_{\theta }(s_{t+2}) - V_{\theta}(s_{t+1})\\\\
$$
带入 $(8)$
$$
A^1_{\theta}(s_t, a) = \delta _t\\\\
A^2_{\theta}(s_t, a) = \delta _t + \gamma \delta _{t+1} \\\\
\cdots 
$$
显然采样次数越多，估计的 $A_{\theta}(s, a)$ 的偏差越小，但方差越大。这需要我们进行权衡，到底选择采样几次作为优势函数。

而 GAE(Generallized Advantage Estimation) 的做法是：所有 $A^k_{\theta}$ 都要，但给每个分配一个权重系数：
$$
A^{GAE} _{\theta} (s, t) = (1-\lambda)(A^1_{\theta} + \lambda \cdot A^2_\theta + \lambda ^2 \cdot A^3_\theta + \cdots)\tag{9}
$$
例如 $\lambda = 0.9$：
$$
A^{GAE} = 0.1 A^1 + 0.09 A^2 + 0.081 A^3 + \cdots
$$
当前这个不会无限下去，因为最后某个 $T$ 时刻路径会结束。GAE 就是从当前 $t$ 直接一直算到最后。

最后带入梯度：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta} (s_{t}^n, a_{t}^n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{10}
$$


## On Policy / Off Policy

On Policy：使用 Agent 到环境中执行得到回报和动作概率，根据 $10$ 更新参数。这样模型跟环境交互一次，就更新模型参数，之后就要继续交互。所以更新的模型和系统交互的模型是同一个参数，就是 On Policy。

Off Policy 希望：用一个模型和环境交互，得到奖励和概率，而另一个模型根据信息更新自己的参数，更新参数的模型并没有去环境交互。这样交互一次，可以给多个模型更新，提高训练效率。

这需要使用重要性采样：
$$
E(f(x))_{x \sim p(x)} = \sum_x f(x) p(x) 
\\\\ = \sum_x f(x) p(x) \frac{q(x)}{q(x)}
\\\\ = \sum_x \left( f(x) \frac{p(x)}{q(x)}\right ) q(x)
\\\\ = E(f(x)\frac{p(x)}{q(x)})_{x \sim q(x)}
\\\\ \approx \frac 1 N \sum_{n=1}^N f(x)\frac{p(x)}{q(x)}
$$
即：我们想计算 $p(x)$ 下的期望，可以通过计算 $q(x)$ 下的期望，并且加上一个比率 $\frac{p(x)}{q(x)}$ 得到。

对模型的训练来说，原始的梯度为：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta} (s_{t}^n, a_{t}^n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{10}
$$
这本质上也是一个期望的形式，我们在 $\theta$ 参数的部分下，计算 $\sum _{t=1}^T A^{GAE}_{\theta} (s_{t}^n, a_{t}^n) \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)$。

现在准备一个参考模型 $\theta'$。让这个模型进行交互，得到 $A^{GAE}_{\theta'} (s_{t}^n, a_{t}^n)$ 和 $P_{\theta'}(a_{t}^n \mid s_{t}^n)$：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta'} (s_{t}^n, a_{t}^n) \frac{P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)}  \nabla\ln P_{\theta}(a_{t}^n \mid s_{t}^n)\tag{11}
$$
展开 $ln$ 的求导：
$$
\frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta'} (s_{t}^n, a_{t}^n) \frac{P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)}  \frac {\nabla P_{\theta}(a_{t}^n \mid s_{t}^n)} {P_{\theta}(a_{t}^n \mid s_{t}^n)}
\\\\
= \frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta'} (s_{t}^n, a_{t}^n) \frac{\nabla P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)} \tag{12}
$$


## KL 散度

虽然使用了参考模型进行 Off Policy，但有一个限制，这个 $\theta'$ 的行为不可以距离 $\theta$ 太远！使用 KL 散度作为 loss 约束。
$$
Loss = - \frac 1 N \sum_{n=1}^N  \sum _{t=1}^T A^{GAE}_{\theta'} (s_{t}^n, a_{t}^n) \frac{ P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)} \tag{13} + \beta KL(P_{\theta} ,P_{\theta'})
$$
同时为了避免模型更新太快，给重要性采样部分增加一个 $clip$：
$$
Loss = -\frac{1}{N} \sum_{n=1}^N \sum_{t=1}^T A^{GAE}_{\theta'}(s_t^n, a_t^n)\min \left( \frac{ P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)}, clip(\frac{ P_{\theta}(a_{t}^n \mid s_{t}^n)}{P_{\theta'}(a_{t}^n \mid s_{t}^n)}, 1-\epsilon, 1 + \epsilon)  \right) + \beta KL(P_{\theta} ,P_{\theta'})
$$
