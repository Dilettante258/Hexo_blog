---
title: 离线强化学习-CQL
categories: 强化学习 离线强化学习
date: 2023-11-9 15:30:23
mathjax: true
---

arxiv: [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)

作者的代码:[CQL-Github](https://github.com/aviralkumar2907/CQL)

> Effectively leveraging large, previously collected datasets in reinforcement learn- ing (RL) is a key challenge for large-scale real-world applications. Offline RL algorithms promise to learn effective policies from previously-collected, static datasets without further interaction. However, in practice, offline RL presents a major challenge, and standard off-policy RL methods can fail due to overestimation of values induced by the distributional shift between the dataset and the learned policy, especially when training on complex and multi-modal data distributions.

如何在强化学习（RL）中有效利用以前收集的大型数据集是大规模在现实世界应用所面临的关键挑战。离线RL算法允许了从以前收集的静态数据集中学习有效的策略，而无需与环境交互。然而，在实践中，离线RL一个主要的挑战是：标准的off-policy RL算法可能会因为数据集和学习策略之间的 distributional shift 导致V值估计过高而失效，尤其是在对复杂和多模态数据分布进行训练时。

> In this paper, we propose conservative Q-learning (CQL), which aims to address these limitations by learning a conservative Q-function such that the expected value of a policy under this Q-function lower-bounds its true value.  ...... If we can instead learn a conservative estimate of the value function, which provides a lower bound on the true values, this overestimation problem could be addressed.

Aviral Kumar等人提出了保守Q学习（CQL），其目的是通过学习一个保守的Q函数来解决这些限制，使得在该Q函数下策略的期望值低于其真实值。 …… 如果我们能够学习值函数的保守估计，它反映了真实Q值的下界，那么这个高估问题就可以得到解决。



> In practice, CQL augments the standard Bellman error objective with a simple Q-value regularizer which is straightforward to implement on top of existing deep Q-learning and actor-critic implementations. 

在实践中，CQL用一个简单的Q值正则化项增强了Bellman error objective的性能，这个修改在现有的deep Q-learning和Actor-Critic算法基础之上都很易于实现。（只需修改大概20行代码）

> Directly utilizing existing value-based off-policy RL algorithms in an offline setting generally results in poor performance, due to issues with bootstrapping from out-of-distribution actions and overfitting. If we can instead learn a conservative estimate of the value function, which provides a lower bound on the true values, this overestimation problem could be addressed.

在离线强化学习环境中直接使用现有的基于值的 off-policy 算法通常会导致较差的性能，这是由于OOD的actions 评估（源于收集数据的策略和学习的策略之间的distributional shift）和自举带来的过拟合问题。如果我们能够学习值函数的保守估计(真值下界)，那么这个高估问题就可以得到解决。

> In fact, because policy evaluation and improvement typically only use the value of the policy, we can learn a less conservative lower bound Q-function, such that only the expected value of Q-function under the policy is lower-bounded, as opposed to a point-wise lower bound.

因为实际上策略评估和改进通常只使用策略的值，所以我们可以学习一个不太保守的Q函数下界，这样该策略下Q函数是期望意义上的下界，而不是逐点的下界（允许某些点不是下界）。

> the key idea behind our method is to minimize values under an appropriately chosen distribution over state-action tuples, and then further tighten this bound by also incorporating a *maximization* term over the data distribution.

我们的方法背后的关键思想是在适当的一些状态-动作组 (s,a) (数据集外的）上最小化值，然后再通过在引入*最大化项*（在数据集内的）来进一步收紧这一界限。



# Preliminaries

> The goal in reinforcement learning is to learn a policy that maximizes the expected cumulative discounted reward in a Markov decision process (MDP)。

强化学习的目标是在马尔可夫决策过程（MDP）中学习使预期累积折扣回报最大化的策略。

D is the dataset, and $d^{\pi_{\beta}}(\mathbf{s})$ is the discounted marginal state-distribution of $\pi_\beta(\mathbf{a}|\mathbf{s})$.

> 边际先验分布（Marginal Prior Distribution）是在没有考虑其他参数的条件下对单个参数的分布进行建模。具体来说，假设我们有一个参数向量θ，其中包含多个参数。边际先验分布是指我们只对其中一些参数感兴趣，而对其他参数不关心。因此，通过将其他参数积分或边际化掉，我们得到一个边际先验分布，其中包含我们关心的参数。

如果对于论文和文章中的符号有不明白的，请参阅这篇文章[CQL1(下界Q值估计)](https://blog.csdn.net/lvoutongyi/article/details/129754201)

> Off-policy RL algorithms based on dynamic programming maintain a parametric Q-function $Q_\theta(s, a)$ and, optionally, a parametric policy, $\pi_{\phi}(a|s)$. Q-learning methods train the Q-function by iteratively applying the Bellman optimality operator.

基于动态规划的Off-policy 算法保持参数Q函数 $Q_\theta(s, a)$，以及可选的参数策略 $\pi_{\phi}(a|s)$。

Q学习方法通过迭代应用Bellman最优算子来训练Q函数。

#### Bellman 最优算子(QL)与Bellman算子(AC)

Bellman 最优算子为Q-Learning(QL)更新时候采用的Q值更新方式，称之为$B^{*}$，定义如下,其中 γ 为折扣因子(discounted-factor)：
$$
B^*Q(s_t,a_t)=r(s_t,a_t)+\gamma E_{s_{t+1}\sim_T}[max_aQ(s_t,a)]
$$
Bellman算子为Actor-Critic(AC)更新时候采用的Q值更新方式，称之为$B^{\pi}$，定义如下：
$$
B^{\pi}Q(s_{t},a_{t})=r(s_{t},a_{t})+\gamma E_{s_{t+1}}\sim_{T,a_{t+1}}\sim_{\pi}[Q(s_{t+1},a_{t+1})]
$$



> In an actor-critic algorithm, a separate policy is trained to maximize the Q-value. Actor-critic methods alternate between computing $Q^\pi$ via (partial) policy evaluation, by iterating the Bellman operator $\mathcal{B}^{\pi}Q=r+\gamma P^{\pi}Q$, where $P^{\pi}$ is the transition matrix coupled with the policy, and improving the policy π(a|s) by updating it towards actions that maximize the expected Q-value.

在Actor-Critic算法中，训练单独的策略以最大化Q值。Actor-critic方法在通过（部分）策略评估计算$Q^\pi$之间交替，方法是迭代Bellman算子$\mathcal{B}^{\pi}Q=r+\gamma P^{\pi}Q$（其中$P^{\pi}$是与策略耦合的状态转换矩阵），以及改进策略π(a|s)，更新以最大化期望Q值。
$$
{\small \begin{aligned}&\hat{Q}^{k+1}\leftarrow\arg\min_Q\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}\left[\left((r(\mathbf{s},\mathbf{a})+\gamma\mathbb{E}_{\mathbf{a}^{\prime}\sim\hat{\pi}^k(\mathbf{a}^{\prime}|\mathbf{s}^{\prime})}[\hat{Q}^k(\mathbf{s}^{\prime},\mathbf{a}^{\prime})])-Q(\mathbf{s},\mathbf{a})\right)^2\right]\text{(policy evaluation)}\\&\hat{\pi}^{k+1}\leftarrow\arg\max_{\pi}\mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim\pi^k(\mathbf{a}|\mathbf{s})}\left[\hat{Q}^{k+1}(\mathbf{s},\mathbf{a})\right]\quad\text{(policy improvement)}\end{aligned}}
$$
第一行中的方程等价于以下方程，证明：[link](https://blog.csdn.net/lvoutongyi/article/details/129754201#:~:text=%E5%AE%9A%E7%90%861%3A-,%E4%B8%8B%E4%B8%A4%E4%B8%AABellman%E4%BC%98%E5%8C%96%E5%BC%8F%E7%AD%89%E4%BB%B7,-(%201%20)%20Q%20k )
$$
Q^{k+1}(s,a)\leftarrow r(s,a)+\gamma E_{s^{\prime}\sim T,a^{\prime}\sim\pi}[Q^{k}(s^{\prime},a^{\prime})]
$$
但是显然的在Offline RL中存在这样的问题，上述公式中的 **r(s,a)** 在Offline中是无法获取的，Since无法与环境进行探索，策略 π 被由于价值最大化操作，价值估计很可能就会偏向于错误的高Q值的 out-of-distribution (OOD) 行为。

# The Conservative Q-Learning (CQL) Framework

离线强化学习面对的巨大挑战是如何减少外推误差。实验证明，外推误差主要会导致在远离数据集的点上Q函数的过高估计，甚至常常出现Q值向上发散的情况。因此，如果能用某种方法将算法中偏离数据集的点上的Q函数保持在很低的值，或许能消除部分外推误差的影响，这就是**保守 Q-learning**（conservative Q-learning，CQL）算法的基本思想。CQL 在普通的贝尔曼方程上引入一些额外的限制项，达到了这一目标。

> develop a conservative Q-learning algorithm, such that the expected value of a policy under the learned **Q-function lower-bounds its true value(真实Q值函数下界)**. Lower-bounded Q-values prevent the over-estimation that is common in offline RL settings due to OOD actions and function approximation error.

CQL算法使策略在所学习的**Q函数的期望值低于其 真值(真实Q值函数下界)**。Q值函数下界防止了由于OOD动作和函数近似误差而在 offline RL 环境中常见的过度估计。

## Conservative Off-Policy Evaluation

普通 DQN 类方法通过优化 stander Bellman error objective 来更新  Q 价值
$$
\hat{Q}^{k+1}\leftarrow\operatorname{argmin}_Q\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\left(Q(s,a)-\hat{\mathcal{B}}^\pi\hat{Q}^k(s,a)\right)^2\right]
$$

> Because we are interested in preventing overestimation of the policy value, we learn a conservative, lower-bound Q-function by additionally minimizing Q-values alongside a standard Bellman error objective. Our choice of penalty is to minimize the expected Q-value under a particular distribution of state-action pairs, µ(s, a).

由于我们想要预防高估价值函数，我们可以通过在保留 标准的 Bellman error objective 的同时，额外最小化Q值来学习保守的Q函数下界。我们选择的惩罚项是希望在某个特定分布µ(s, a)上的期望Q值最小。

又考虑到 $\hat{\mathcal{B}}^{\pi}$的计算中 $s,a,s'$，都在 D 中，只有 a' 是生成的可能 OOD，因此我们限制 $\mu(s,a)$中的 s 边缘分布为 $d^{\pi_\beta}(s)$，从而有$ \mu(s,a)=d^{\pi_\beta}(s)\mu(a|s)$，迭代更新过程可以表示为：<a name="Equation 1">Equation 1</a>

$$
\hat{Q}^{k+1}\leftarrow\arg\min_Q\left.\alpha\right.\mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim\mu(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]+\frac12\mathbb{E}_{\mathbf{s},\mathbf{a}\sim\mathcal{D}}\left[\left(Q(\mathbf{s},\mathbf{a})-\hat{\mathcal{B}}^{\pi}\hat{Q}^k(\mathbf{s},\mathbf{a})\right)^2\right]
$$

$\alpha$是平衡因子，用来控制两个优化目标的比重。

论文中Theorem 3.1 证明了对任意 $\mu(a|s)$,当 $\alpha$ 足够大时，迭代的收敛结果$\hat{Q}_{\pi}:=\lim_{k\to\infty}\hat{Q}^k$ 会对每一个状态动作对 $(s,a)$ 都形成真实值的下界，即 $\hat{Q}_{\pi}\leq Q^{\pi}(s,a)$。[这里是**point-wise**的下界]

> we can substantially tighten this bound if we are only interested in estimating $V_\pi (s)$. If we only require that the expected value of the $Q^\pi$ under π(a|s) lower-bound $V\pi$, we can improve the bound by introducing an additional Q-value maximization term under the data distribution, $\pi_\beta(a|s)$, resulting in the iterative update (changes in red):

为了防止过于保守，如果我们只对估计只要求$\hat{Q}_\pi$关于策略$\pi(a|s)$的期望$\hat{V}^\pi(s)$感兴趣，我们可以大大收紧这个界（放松对Q的约束，也就是**期望意义上的下界**，**允许某些点不是下界**）。对于符合用于生成数据集 $\mathcal{D}$ 的行为策略 π  的数据点，我们可以认为对这些点的估值较为准确，在这些点上不必限制让值很小，我们可以通过引入**对在数据集分布$\pi_\beta(a|s)$上的Q值进行最大化**的项来改进(最小化负值=最大化原值)，从而得出（红色标记变化）：<a name="Equation 2">Equation 2</a>
$$
\begin{aligned}\hat{Q}^{k+1}\leftarrow\arg\min_Q\alpha\cdot\left(\mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim\mu(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]-{\color{Red} \mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim\hat{\pi}_{\beta}(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]} \right)\\+\frac{1}{2}\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}\left[\left(Q(\mathbf{s},\mathbf{a})-\hat{\mathcal{B}}^\pi\hat{Q}^k(\mathbf{s},\mathbf{a})\right)^2\right]\end{aligned}
$$

其中$\hat{\pi}_\beta$ 是利用数据集 $D$ 得到的对真实行为策略 $\pi_\beta$ 的估计，因为我们无法获知真实的行为策略，只能通过数据集中已有的数据近似得到。

所以那些在行为策略当中的动作**就有可能被高估**。

论文中Theorem 3.2 证明了当 $\mu(a|s)=\pi(a|s)$ 时，上式迭代收敛得到的$Q$ 函数虽然不是在每一点上都小于真实值，但其期望是小于真实值的，即 $\mathbb{E}_{\pi(a|s)}\left[\hat{Q}^{\pi}(s,a)\right]=\hat{V}^{\pi}(s)\leq V^{\pi}(s)$。[这里**不是point-wise**的下界]

> In summary, we showed that the basic CQL evaluation in <a href="#Equation 1">Equation 1</a> learns a Q-function that lower-bounds the true Q-function Qπ , and the evaluation in <a href="#Equation 2">Equation 2</a>  provides a tighter lower bound on the expected Q-value of the policy π. 

总之，我们发现<a href="#Equation 1">Equation 1</a>中的基本CQL评估学习了一个Q函数，该Q函数下界为真实Q函数$Q^\pi$，而<a href="#Equation2">Equation2</a>的评估为策略π的预期Q值提供了一个更严格的下界。

# Conservative Q-Learning for Offline RL

> We now present a general approach for offline policy learning, which we refer to as conservative Q-learning (CQL). As discussed in Section 3.1, we can obtain Q-values that lower-bound the value of a policy π by solving <a href="#Equation 2">Equation 2</a> with µ = π. How should we utilize this for policy optimization? We could alternate between performing full off-policy evaluation for each policy iterate, $\pi^k$, and one step of policy improvement. However, this can be computationally expensive. Alternatively, since the policy $\pi^k$ is typically derived from the Q-function, we could instead choose µ(a|s) to approximate the policy that would maximize the current Q-function iterate, thus giving rise to an online algorithm.

至此，我们提出了一种离线策略学习的通用方法，称为 保守Q学习（CQL）。但仍有一个缺陷：计算的时间开销太大了。

上面提到，我们可以通过求解<a href="#Equation 2">Equation 2</a>（其中µ = π）来获得Q值，这些Q值构成了策略π的价值的下界。我们应该如何利用这一点进行策略优化呢？当令 $\mu(a|s)=\pi(a|s)$ 时，迭代的每一步，算法都要对策略 $\hat{\pi}^k$ 做完整的离线策略评估（迭代以上更新式至收敛）来计算 $\arg min$，再进行一次策略迭代，而离线策略评估是非常耗时的。作为替代，考虑到策略$\pi^k$通常是从在每轮迭代过程中由使 Q 最大的动作中派生出来的，我们可以选择µ(a|s)来近似 <font color=greenyellow>将使当前Q函数迭代最大化的</font> 策略，从而产生一种在线算法。
$$
\pi\approx\max_{\mu}\mathbb{E}_{s\sim\mathcal{D},a\sim\mu(a|s)}[Q(s,a)]
$$



> We can formally capture such online algorithms by defining a family of optimization problems over µ(a|s), presented below, with modifications from <a href="#Equation 2">Equation 2</a> marked in red. An instance of this family is denoted by CQL(R) and is characterized by a particular choice of regularizer R(µ):

我们可以通过在µ(a|s)上定义一族优化问题来形式化地捕捉这种在线算法，如下所示，下面通过红色标记出来对<a href="#Equation 2">Equation 2</a>进行的修改。这个被称为CQL(R)，为了防止过拟合，再加上正则化项 ${\mathcal{R}(\mu)}$：<a name="Equation 3">Equation 3</a>
$$
\begin{aligned}
\min_Q{\color{red}\max_{\mu}}\alpha\left(\mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim{\color{red}\mu{(\mathbf{a}|\mathbf{s})}}}\left[Q(\mathbf{s},\mathbf{a})\right]-\mathbb{E}_{\mathbf{s}\sim\mathcal{D},\mathbf{a}\sim\hat{\pi}_{\beta}{(\mathbf{a}|\mathbf{s})}}\left[Q(\mathbf{s},\mathbf{a})\right]\right)\\+\frac12\left.\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}{\left[\left(Q(\mathbf{s},\mathbf{a})-\hat{\mathcal{B}}^{\pi_k}\hat{Q}^k(\mathbf{s},\mathbf{a})\right)^2\right]}+\color{red}{\mathcal{R}(\mu)}\right. 
\end{aligned}
$$

### Variants of CQL

> To demonstrate the generality of the CQL family of optimization problems, we discuss two specific instances within this family that are of special interest, and we evaluate them empirically in Section 6. If we choose $\mathcal{R}(\mu)$ to be the KL-divergence against a prior distribution, $\rho(a|s)$, i.e., $\mathcal{R}(\mu)=-D_{\mathrm{KL}}(\mu,\rho)$, then we get $\mu(\mathbf{a}|\mathbf{s})\propto\rho(\mathbf{a}|\mathbf{s})\cdot{\exp(Q(\mathbf{s},\mathbf{a}))}$ (for a derivation, see Appendix A). First, if $\rho=\mathrm{Unif}(a)$, then the first term in Equation 3 corresponds to a soft-maximum of the Q-values at any states and gives rise to the following variant of Equation 3, called CQL(H):

……如果我们将正则化项$\mathcal{R}(\mu)$采用为对先验策略和$\rho(a|s)$ 的 KL散度，即$\mathcal{R}(\mu)=-D_{\mathrm{KL}}(\mu,\rho)$，那么我们可以得到$\mu(\mathbf{a}|\mathbf{s})\propto\rho(\mathbf{a}|\mathbf{s})\cdot{\exp(Q(\mathbf{s},\mathbf{a}))}$。如果取 $\rho(a|s)$ 为均匀分布 $\mathcal{U}(a)$，即 $\rho=\mathrm{Unif}(a)$，那么<a href="#Equation 3">Equation 3</a>中的第一项对应于任意状态s处Q值的软最大值，并引出以下 CQL(R) 的变体，称为CQL(H)：<a name="Equation 4">Equation 4</a>
$$
\min_Q\left.\alpha\mathbb{E}_{\mathbf{s}\sim\mathcal{D}}\left[\log\sum_{\mathbf{a}}\exp(Q(\mathbf{s},\mathbf{a}))-\mathbb{E}_{\mathbf{a}\sim\hat{\pi}_\beta(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]\right]+\frac{1}{2}\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}\left[\left(Q-\hat{\mathcal{B}}^{\pi_k}\hat{Q}^k\right)^2\right]\right.
$$
可以注意到，简化后式中已经不含有 µ，为计算提供了很大方便。

详细推导过程请参见 [动手做强化学习](https://hrl.boyuai.com/chapter/3/离线强化学习/#186-扩展阅读)

论文中 Theorem 3.3 证明了：若策略梯度更新的非常缓慢(足够小的速度更新),不考虑采样误差，即$\hat{B}^\pi=B^\pi$,那么选取$\mu=\hat{\pi}^k$,可以保证在迭代更新中的每一步时，都有$\hat{V}^{k+1}(s)\leq V^{k+1}(s)$

论文中 Theorem 3.4 证明了：CQL是gap-expanding的，即对于任意一次迭代，in-distribution动作分布与OOD动作分布产生的Q函数（$ \hat{Q}^k$）的差将比在真实Q函数上产生的差更大，公式表达是$\mathbb{E}_{\pi_\beta(\mathbf{a}|\mathbf{s})}[\hat{Q}^k(\mathbf{s},\mathbf{a})]-\mathbb{E}_{\mu_k(\mathbf{a}|\mathbf{s})}[\hat{Q}^k(\mathbf{s},\mathbf{a})]>\mathbb{E}_{\pi_\beta(\mathbf{a}|\mathbf{s})}[Q^k(\mathbf{s},\mathbf{a})]-\mathbb{E}_{\mu_k(\mathbf{a}|\mathbf{s})}[Q^k(\mathbf{s},\mathbf{a})]$。本文的方法存在潜在的优势是能够在真实Q函数上缩紧 训练的策略  与行为策略 的差距。换句话表示就是，**学习到的策略**和**行为策略**之间的**正则化Q函数期望估值差异**，都**比原本的Q函数大**。这样的话在面对**函数近似**以及**采样误差**的时候**可以更稳定**。直观来说就是**分布内动作**受到**分布外动作**的**影响更小**。

> Second, if ρ(a|s) is chosen to be the previous policy $\pi^{k−1}$, the first term in <a href="#Equation 4">Equation 4</a> is replaced by an exponential weighted average of Q-values of actions from the chosen πˆ k−1 (a|s). Empirically, we 4 find that this variant can be more stable with high-dimensional action spaces (e.g., Table 2) where it is challenging to estimate $\log\sum_\mathbf{a}{\exp}$ via sampling due to high variance. In Appendix A, we discuss an additional variant of CQL, drawing connections to distributionally robust optimization. We will discuss a practical instantiation of a CQL deep RL algorithm in Section 4. CQL can be instantiated as either a Q-learning algorithm (with $B^*$ instead of $B^\pi$ in Equations 3, 4) or as an actor-critic algorithm.

在实验中，由于高方差，要通过采样估计estimate $\log\sum_{a}exp(Q(s,a))$ 并不容易。如果选择ρ(a|s)为先前的策略$\pi^{k−1}$，那么<a href="#Equation 4">Equation 4</a>中的第一项将被从选择的$\pi^{k−1}(a|s)$的动作的Q值的指数加权平均值所替代。根据实验，这个变种在高维动作空间中可以更加稳定。请见 [CQL论文](https://arxiv.org/pdf/2006.04779.pdf)  附录A   CQL(var)部分。



CQL可以作为Q学习算法（在<a href="#Equation 3">Equation 3</a>、<a href="#Equation 4">4</a>中使用$B^*$而不是$B^\pi$）或作为演员-评论家算法来实现。

### Computing $\log\sum_\mathbf{a}{\exp(Q(\mathbf{s},\mathbf{a})})$

> $\operatorname{CQL}(\mathcal{H})$ uses log-sum-exp in the objective for training the Q-function (Equation 4). In discrete action domains, we compute the log-sum-exp exactly by invoking the standard tf.reduce_logsumexp() (or torch.logsumexp()) functions provided by autodiff libraries. In continuous action tasks, $\operatorname{CQL}(\mathcal{H})$ uses importance sampling to compute this quantity, where in practice, we sampled 10 action samples each at every state s from a uniform-at-random Unif(a) and the current policy, π(a|s) and used these alongside importance sampling to compute it as follows using N = 10 action samples:

$\operatorname{CQL}(\mathcal{H})$在训练Q函数（<a href="#Equation 4">Equation 4</a>）时在目标函数中使用了 log-sum-exp。在离散动作空间中，我们可以通过调用自动微分库提供的标准函数（`tf.reduce_logsumexp()`或`torch.logsumexp()`）来精确计算log-sum-exp。在连续动作任务中，$\operatorname{CQL}(\mathcal{H})$ 使用重要性采样来计算这个值，在实践中，我们从均匀随机分布Unif(a)和当前策略π(a|s)中对每个状态s采样了10个动作样本，并使用重要性采样来计算如下，其中N = 10表示动作样本的数量：


$$
\begin{aligned}
\log\sum_{\mathbf{a}}\exp(Q(\mathbf{s},\mathbf{a}))& =\log\left(\frac12\sum_\mathbf{a}\exp(Q(\mathbf{s},\mathbf{a}))+\frac12\sum_\mathbf{a}\exp(Q(\mathbf{s},\mathbf{a}))\right)  \\
&=\log\left(\frac12\mathbb{E}_{\mathbf{a}\sim\mathrm{Unif}(\mathbf{a})}\left[\frac{\exp(Q(\mathbf{s},\mathbf{a}))}{\mathrm{Unif}(\mathbf{a})}\right]+\frac12\mathbb{E}_{\mathbf{a}\sim\pi(\mathbf{a}|\mathbf{s})}\left[\frac{\exp(Q(\mathbf{s},\mathbf{a}))}{\pi(\mathbf{a}|\mathbf{s})}\right]\right) \\
&\approx\log\left(\frac1{2N}\sum_{\mathbf{a}_i\sim\mathrm{Unif}(\mathbf{a})}^N\left[\frac{\exp(Q(\mathbf{s},\mathbf{a}_i))}{\mathrm{Unif}(\mathbf{a})}\right]+\frac1{2N}\sum_{\mathbf{a}_i\sim\pi(\mathbf{a}|\mathbf{s})}^N\left[\frac{\exp(Q(\mathbf{s},\mathbf{a}_i))}{\pi(\mathbf{a}_i|\mathbf{s})}\right]\right)
\end{aligned}
$$

# CQL 代码实践

## Pseudocode

![](https://img-blog.csdnimg.cn/img_convert/0e3840978ea79f701418226b372dde41.png)

如果是Q-learning模式：μ 可以作为最终的策略
如果是Actor-Critic模式：需要使用SAC的训练方式额外训练actor







![image-20231113111241767](http://106.15.139.91:40027/uploads/2312/658d4d8e486ab.png)

[动手做强化学习](https://hrl.boyuai.com/chapter/3/离线强化学习#184-cql-代码实践)中，CQL代码实践，相较于SAC代码实践主要多的是这一部分：

```python
# 以上与SAC相同,以下Q网络更新是CQL的额外部分
batch_size = states.shape[0]
random_unif_actions = torch.rand([batch_size * self.num_random, actions.shape[-1]],
dtype=torch.float).uniform_(-1, 1).to(device)
random_unif_log_pi = np.log(0.5**next_actions.shape[-1])
print('random_unif_log_pi: ',random_unif_log_pi,np.log(0.5))
tmp_states = states.unsqueeze(1).repeat(1, self.num_random,1).view(-1, states.shape[-1])
tmp_next_states = next_states.unsqueeze(1).repeat(1, self.num_random, 1).view(-1, next_states.shape[-1])
random_curr_actions, random_curr_log_pi = self.actor(tmp_states)
random_next_actions, random_next_log_pi = self.actor(tmp_next_states)
q1_unif = self.critic_1(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
q2_unif = self.critic_2(tmp_states, random_unif_actions).view(-1, self.num_random, 1)
q1_curr = self.critic_1(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
q2_curr = self.critic_2(tmp_states, random_curr_actions).view(-1, self.num_random, 1)
q1_next = self.critic_1(tmp_states, random_next_actions).view(-1, self.num_random, 1)
q2_next = self.critic_2(tmp_states, random_next_actions).view(-1, self.num_random, 1)
q1_cat = torch.cat([
	q1_unif - random_unif_log_pi,
	q1_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
	q1_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
	],dim=1)
q2_cat = torch.cat([
	q2_unif - random_unif_log_pi,
	q2_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),
	q2_next - random_next_log_pi.detach().view(-1, self.num_random, 1)
	],dim=1)

qf1_loss_1 = torch.logsumexp(q1_cat, dim=1).mean()
qf2_loss_1 = torch.logsumexp(q2_cat, dim=1).mean()
qf1_loss_2 = self.critic_1(states, actions).mean()
qf2_loss_2 = self.critic_2(states, actions).mean()
qf1_loss = critic_1_loss + self.beta * (qf1_loss_1 - qf1_loss_2)
qf2_loss = critic_2_loss + self.beta * (qf2_loss_1 - qf2_loss_2)
```

`beta`是CQL损失函数中的系数，`num_random` 是CQL中的动作采样数。

上面这段代码的意义是：

1. 生成一个随机动作 `random_unif_actions`，其形状与输入动作 `actions` 相同，但是值是在区间 `[-1, 1]` 内均匀分布的随机数。`random_unif_actions: torch.Size([320, 1])`
2. 计算一个常数 `random_unif_log_pi`，它是以 0.5 为底的 `next_actions` 张量的维度数的对数。`random_unif_log_pi: int`
3. 对输入状态 `states` 和 `next_states` 进行扩展，以便与 `random_unif_actions` 的维度匹配，从而得到 `tmp_states` 和 `tmp_next_states`。`tmp_states: torch.Size([320, 3])`</br> 
    这段代码的作用是将`states`张量进行重复和重新形状操作：
    1. `states.unsqueeze(1)`：在维度1上添加一个维度，这样做是为了在后续操作中能够正确地重复`states`张量。`torch.Size([64, 1, 3])`
    2. `.repeat(1, self.num_random, 1)`：将`states`张量在维度1上重复`self.num_random`次（动作采样数），即，得到形状为`(batch_size, self.num_random)`的张量。这样做是为了在后续操作中生成多个重复的状态。`torch.Size([64, 5, 3])`
    3. `.view(-1, states.shape[-1])`：将张量重新形状为`(batch_size * self.num_random / 3, 3)`，其中`-1`表示自动计算该维度的大小。这样做是为了将重复的状态展平为一个二维张量，以便进行后续处理。

4. 使用 `self.actor` 函数计算 `tmp_states` 和 `tmp_next_states` 对应的随机动作以及它们的对数概率。`random_curr_actions, random_curr_log_pi: torch.Size([320, 1])`此处的states来自于训练集(buffer)中，将该state输入到actor中，因为是连续的动作值，通过高斯分布采样得到random_curr_actions, 以及可以得到对应的log_pi。
5. 现在我们就到了采样的action以及对应的log_prob，分别来自于均匀分布、当前s的高斯分布以及下一个s的高斯分布。我们下一步需要得到Q(s,a)，也就是需要限制的q。得到的(s,a) 对输入到critic中就行了。使用 `self.critic_1` 和 `self.critic_2` 函数计算 `tmp_states` 和随机动作 `random_unif_actions`、`random_curr_actions`、`random_next_actions` 对应的 Q 值。`q1: torch.Size([64, 5, 1])`
6. 将 Q 值按照一定的方式进行组合，得到 `q1_cat` 和 `q2_cat`。这些组合包括将随机动作的 Q 值减去对应的对数概率，并将其与其他 Q 值拼接在一起。`q_cat: torch.Size([64, 15, 1])`
7. 计算`qf_loss_1`：将`q_cat`指数之和的对数压缩到一维(`torch.Size([64, 1]`)，再求平均。
    $$
    \mathrm{logsumexp}(x)_i=\log\sum_j\exp(x_{ij})
    $$
    这样就构成了：
$$
\min_Q\left.\alpha\mathbb{E}_{\mathbf{s}\sim\mathcal{D}}\left[\log\sum_{\mathbf{a}}\exp(Q(\mathbf{s},\mathbf{a}))-\mathbb{E}_{\mathbf{a}\sim\hat{\pi}_\beta(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]\right]+\frac{1}{2}\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}\left[\left(Q-\hat{\mathcal{B}}^{\pi_k}\hat{Q}^k\right)^2\right]\right.
$$

`critic_1_loss`是 $\frac{1}{2}\mathbb{E}_{\mathbf{s},\mathbf{a},\mathbf{s}^{\prime}\sim\mathcal{D}}\left[\left(Q-\hat{\mathcal{B}}^{\pi_k}\hat{Q}^k\right)^2\right]$ ，`self.beta`是公式中的 $\alpha$ ，`qf1_loss_1 - qf1_loss_2`即是 $\mathbb{E}_{\mathbf{s}\sim\mathcal{D}}\left[\log\sum_{\mathbf{a}}\exp(Q(\mathbf{s},\mathbf{a}))-\mathbb{E}_{\mathbf{a}\sim\hat{\pi}_\beta(\mathbf{a}|\mathbf{s})}\left[Q(\mathbf{s},\mathbf{a})\right]\right]$ 。

| 变量名                  | 意义                                                    | 来源                                             |     Size      |
| :---------------------- | :------------------------------------------------------ | :----------------------------------------------- | :-----------: |
| states                  | 状态                                                    | Replay Buffer                                    |    [64, 3]    |
| batch_size              | 批次大小                                                | states                                           |      64       |
| random_unif_actions     | 随机均匀采样的动作                                      | torch.rand().uniform_(-1, 1)                     |   [64*5, 1]   |
| random_unif_log_pi      | 均匀分布的对数概率                                      | np.log(0.5)                                      | numpy.float64 |
| tmp_states              | 扩展后的状态                                            | states                                           |   [320, 3]    |
| tmp_next_states         | 扩展后的下一个状态                                      | next_states                                      |   [320, 3]    |
| random_curr_actions     | 当前动作                                                | tmp_states                                       |   [320, 1]    |
| random_curr_log_pi      | 当前动作的对数概率                                      | tmp_states                                       |   [320, 1]    |
| random_next_actions     | 下一个动作                                              | tmp_next_states                                  |   [320, 1]    |
| random_next_log_pi      | 下一个动作的对数概率                                    | tmp_next_states                                  |   [320, 1]    |
| q1_unif / q2_unif       | 使用随机均匀采样动作计算Critic的的Q值                   | tmp_states, random_unif_actions                  |  [64, 5, 1]   |
| q1_curr / q2_curr       | 使用随机当前动作计算的第一个Critic的Q值                 | tmp_states, random_curr_actions                  |  [64, 5, 1]   |
| q1_next / q2_next       | 使用随机下一个动作计算的Critic的Q值                     | tmp_states, random_next_actions                  |  [64, 5, 1]   |
| q1_cat / q2_cat         | 拼接后的Critic的Q值                                     | ......                                           |  [64, 15, 1]  |
| qf1_loss_1 / qf2_loss_1 | Critic的损失函数（q_cat的对数求和的平均值）             | q1_cat / q2_cat                                  |  FloatTensor  |
| qf1_loss_2 / qf2_loss_2 | Critic的损失函数（使用原始状态和动作计算的Q值的平均值） | states, actions                                  |  FloatTensor  |
| qf1_loss / qf2_loss     | Critic的总损失函数                                      | critic_1_loss + beta * (qf1_loss_1 - qf1_loss_2) |  FloatTensor  |



Reference：

1. [Conservative Q-Learning(CQL)保守Q学习(一)-CQL1(下界Q值估计)](https://blog.csdn.net/lvoutongyi/article/details/129754201
    )
2. [Conservative Q-Learning(CQL)保守Q学习(二)-CQL2(下界V值估计),CQL(R)与CQL(H)](https://blog.csdn.net/lvoutongyi/article/details/129780619)（这两篇理论证明过程很详细，缺少对后几个Theorem的详细解释）
3. [论文速览【Offline RL】—— 【CQL】Conservative Q-Learning for Offline Reinforcement Learning](https://blog.csdn.net/wxc971231/article/details/131588429)（很精简）
4. [CQL: Conservative Q-Learning for Offline Reinforcement Learning](https://zhuanlan.zhihu.com/p/633549377) （解释很均衡）
5. [强化学习 | CQL：Conservative Q-Learning for Offline Reinforcement Learning](https://zhuanlan.zhihu.com/p/517608562)（这篇对几个Theorem解释较好）
6. [Conservative Q Learning(保守强化学习)傻瓜级讲解和落地教程](https://zhuanlan.zhihu.com/p/603691759)
7. [【论文笔记】Conservative Q-Learning](https://zhuanlan.zhihu.com/p/429378041)