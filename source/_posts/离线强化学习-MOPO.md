---
title: 离线强化学习-MOPO
categories: 强化学习 离线强化学习
date: 2023-11-12 12:30:23
mathjax: true
---



内容未完成。

内容未完成。

内容未完成。



目前主流的offline RL的方法都是model free的，这类方法通常需要将policy限制到data覆盖的集合范围里（support），不能泛化到没见过的状态上。作者提出Model-based Offline Policy Optimization (MOPO)算法,用model based的方法来做offline RL，同时通过给reward添加惩罚项（soft reward penalty）来描述环境转移的不确定性（applying them with rewards artificially penalized by the uncertainty of the dynamics.）这种方式相当于在泛化性和风险之间做tradeoff。作者的意思是，这种方式允许算法为了更好的泛化性而承担一定风险（policy is allowed to take a few risky actions and then return to the confident area near the behavioral distribution without being terminated）。具体做法就是，先根据data去学一堆状态转移函数，这个函数是一个用神经网络表示的关于状态和reward的高斯分布

MOPO: Model-based Offline Policy Optimization

离线强化学习（Offline reinforcement learning, RL）指的是完全从一批先前收集的数据中学习策略的问题。尽管无模型离线RL取得了显著进展，但最成功的现有方法将策略限制在数据支持范围内，无法推广到新状态。
在本文中，作者观察到，与无模型方法相比，现有的基于模型的RL算法本身已经在离线设置中产生了显著的优势，尽管它并不是为这种设置而设计的。然而，尽管许多基于标准模型的RL方法已经估计了模型的不确定性，但它们本身并没有提供避免与离线环境中的分布偏移相关的问题的机制。因此，作者建议修改现有的基于模型的RL方法，以通过将基于模型的脱机RL转换为受惩罚的MDP框架来解决这些问题。
作者从理论上表明，通过使用这种惩罚性MDP，可以使真实MDP中的收益下限最大化。基于上述理论结果，作者提出了一种新的基于模型的离线RL算法，该算法将Lipschitz正则化模型的方差作为对奖励函数的惩罚。实验表明，该算法的性能优于现有的离线RL基准测试中基于标准模型的RL方法和现有的最新无模型离线RL方法，以及两项具有挑战性的需要概括从不同任务收集的数据连续控制任务。



MOPO是一种model-based offline RL方法，简单来说就是把MBPO用在了offline设定下，根据offline的需要做了一些小修改。

一方面，这种model-based方法生成 D_model 数据集，能够提升sample efficiency；另一方面，MOPO能够利用model让策略以一定概率选择行为分布之外的action。但是离行为分布越远，不确定性会越大，所以MOPO通过不确定性估计来量化偏离行为分布的风险，实现其和探索多样state所得收益之间的trade-off。具体来说，就是在reward函数后面加了一个不确定性惩罚项，通过ensemble的方法（即N个dynamics模型的最大标准差）量化这种不确定性，不确定性越大则reward越小。也就是说不是直接惩罚OOD状态和动作，而是惩罚具有高不确定性的 (s,a) 。





> To approach this question, we first hypothesize that model-based RL methods  make a natural choice for enabling generalization, for a number of reasons. First, model-based RL algorithms effectively receive more supervision, since the model is trained on every transition, even in sparse-reward settings. Second, they are trained with supervised learning, which provides more stable and less noisy gradients than bootstrapping. Lastly, uncertainty estimation techniques, such as bootstrap ensembles, are well developed for supervised learning methods and are known to perform poorly for value-based RL methods.

为了回答这个问题，我们首先假设模型驱动的强化学习（RL）方法在实现泛化方面具有自然优势，原因如下。首先，模型驱动的RL算法能够更有效地进行监督学习，因为模型在每个转换中都接受训练，即使在稀疏奖励的情况下也是如此。其次，它们通过监督学习进行训练，这提供了比自举法更稳定且噪声更少的梯度。最后，不确定性估计技术（例如自举集成）在监督学习方法中得到了充分发展，而在基于值的RL方法中已知表现不佳。

> All of these attributes have the potential to improve or control generalization. As a proof-of-concept experiment, we evaluate two state-of-the-art off-policy model-based and model-free algorithms, MBPO and SAC, in Figure 1. Although neither method is designed for the batch setting, we find that the model-based method and its variant without ensembles show surprisingly large gains. This finding corroborates our hypothesis, suggesting that model-based methods are particularly well-suited for the batch setting, motivating their use in this paper.

所有这些特性都有潜力改善或控制泛化。作为一个概念验证实验，我们在图1中评估了两种最先进的离策略模型驱动和模型无关算法，MBPO和SAC。尽管这两种方法都不是为批处理设置而设计的，但我们发现模型驱动方法及其不使用集成的变体显示出令人惊讶的巨大增益。这一发现支持了我们的假设，表明模型驱动方法特别适合于批处理设置，并激发了我们在本文中使用它们的动机。



Despite these promising preliminary results, we expect significant headroom for improvement.
In particular, because offline model-based algorithms cannot improve the dynamics model using
additional experience, we expect that such algorithms require careful use of the model in regions
outside of the data support. Quantifying the risk imposed by imperfect dynamics and appropriately
trading off that risk with the return is a key ingredient towards building a strong offline model-based
RL algorithm. To do so, we modify MBPO to incorporate a reward penalty based on an estimate of
the model error. Crucially, this estimate is model-dependent, and does not necessarily penalize all
out-of-distribution states and actions equally, but rather prescribes penalties based on the estimated
magnitude of model error. Further, this estimation is done both on states and actions, allowing
generalization to both, in contrast to model-free approaches that only reason about uncertainty with
respect to actions.



尽管有这些令人鼓舞的初步结果，但我们预计还有很大的改进空间。特别是，由于基于离线模型的算法无法使用额外的经验来改进动力学模型，因此我们预计此类算法需要在数据支持之外的区域谨慎使用模型。量化不完美动态带来的风险，并适当地将风险与回报进行权衡，是构建强大的基于离线模型的RL算法的关键因素。为此，我们修改了 MBPO，以纳入基于模型误差估计的奖励惩罚。至关重要的是，此估计依赖于模型，并且不一定平等地惩罚所有分布外状态和操作，而是根据估计的模型误差大小规定惩罚。此外，这种估计是针对状态和动作进行的，允许对两者进行泛化，这与仅推理动作不确定性的无模型方法形成鲜明对比。



 we develop a practical method that estimates model error using the predicted variance
of a learned model, uses this uncertainty estimate as a reward penalty, and trains a policy using
MBPO in this uncertainty-penalized MDP. 

我们开发了一种实用的方法，该方法使用学习模型的预测方差来估计模型误差，将这种不确定性估计用作奖励惩罚，并在这种不确定性惩罚的MDP中使用MBPO训练策略。



Uncertainty quantification, a key ingredient to
our approach, is critical to good performance in model-based RL both theoretically [ 63 , 75 , 44 ] and
empirically [ 12, 7, 50, 39, 8], and in optimal control [62, 2, 34]. Unlike these works, we develop and
leverage proper uncertainty estimates that particularly suit the offline setting.





>强化学习中的dynamics是什么意思？[^安道龙的回答 - 知乎]
>
>1. Dynamics，动力学，动力学模式，动力学模型
>2. 该词主要出现在强化学习的Model-base类论文中。
>3. 指的是，在强化学习中，环境实际的运行机制不叫做模型，而叫做环境动力学（dynamics of environment），它能够明确确定Agent下一个状态和所得的即时奖励。
>4. 所以动力学指的就是环境运转的动态机制，能够给出状态转移函数和奖励。
>5.  然后Model-Based的强化学习就是要基于交互信息学习一个环境动力学模型（Dynamics Model），并利用该模型进行辅助训练。

为表述统一方便，这里 dynamics 统一译为 动力学模式。





| Symbol                                              |                                                              |                                                              |
| --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| $M=(\mathcal{S},\mathcal{A},T,r,\mu_{0},\gamma)$    | Markov decision process                                      | 马尔可夫决策过程                                             |
| $\mathcal{S}$                                       | state space                                                  | 状态空间                                                     |
| $\mathcal{A}$                                       | action space                                                 | 动作空间                                                     |
| $T(s^{\prime}\mid s,a)$                             | transition dynamics                                          | 转移函数                                                     |
| r(s, a)                                             | reward function                                              | 奖励函数                                                     |
| $μ_0$                                               | initial state distribution                                   | 初始状态分布                                                 |
| γ                                                   | discount factor                                              | 折扣因子                                                     |
| π(a \|s)                                            | policy                                                       | 策略                                                         |
| $η_M(\pi)$                                          | expected discounted return                                   | 预测的总奖励函数                                             |
| $V_{M}^{\pi}(s)$                                    | value function                                               | 价值函数                                                     |
| $\mathcal{D}_{\mathrm{env}}=\{(s,a,r,s^{\prime})\}$ | static dataset                                               | 静态数据集                                                   |
| $\pi^B$                                             | behavior policies                                            | 行为策略                                                     |
| $T$                                                 | the real dynamics                                            | 真实的动态学模型                                             |
| $\widehat{T}$                                       | dynamics model, the learned dynamics                         | 学习的动态学模型，根据$\mathcal{D}$估计转移模型              |
| $\mathbb{P}_{\widehat{T},t}^{\pi}(s)$               | the probability of being in state s at time step t if actions are sampled according to π and transitions according to | 动态学模型$\widehat{T}$下采取策略π使得智能体在时刻t状态为s的概率 |
| $\rho_{\widehat{T}}^{\pi}(s,a)$                     | discounted occupancy measure of policy π under dynamics      | **占用度量**，动作状态对被访问到的概率                       |
| $\widehat{T}_{\theta}(s^{\prime}|s,a)$              | a model of the transition distribution                       | 用参数θ表示的状态转移模型                                    |
| $G_{\widehat{M}}^{\pi}(s,a)$                        | measures the difference between M and Mc under the test function V π |                                                              |
|                                                     |                                                              |                                                              |
|                                                     |                                                              |                                                              |

# MOPO: Model-Based Offline Policy Optimization

our goal is to design an offline model-based reinforcement learning algorithm that can take actions that are not strictly within the support of the behavioral distribution. Using a model gives us the potential to do so.  However, models will become increasingly inaccurate further from the behavioral distribution, and vanilla model-based policy optimization algorithms may exploit these regions where the model is inaccurate. This concern is especially important in the offline setting, where mistakes in the dynamics will not be corrected with additional data collection.

我们的目标是设计一种离线模型驱动的强化学习算法，可以采取不严格符合行为分布的动作。使用模型给予我们这样的潜力。然而，模型在远离行为分布的地方会变得越来越不准确，而传统的模型驱动策略优化算法可能会利用这些模型不准确的区域。在离线设置中，这个问题尤为重要，因为动力学的错误不会通过额外的数据收集来进行纠正。









我们的目标是设计一种离线模型驱动的强化学习算法，可以采取不严格符合行为分布的动作。使用模型给予我们这样的潜力。然而，模型在远离行为分布的地方会变得越来越不准确，而传统的模型驱动策略优化算法可能会利用这些模型不准确的区域。在离线设置中，这个问题尤为重要，因为动力学的错误不会通过额外的数据收集来进行纠正。



## Quantifying the uncertainty: from the dynamics to the total return

Our key idea is to build a lower bound for the expected return of a policy π under the true dynamics and then maximize the lower bound over π.



$\eta_{\widehat{M}}(\pi)$ 是 $\eta_{M}(\pi)$ 对真实回报的估计值，





### Telescoping lemma

假设 $M$ 和 $\widehat{M}$ 是两个有相同奖励函数 r 的 MDP，但是分别有两个不同的 动力学模型 $T$ 和 $\widehat{T}$。

我们定义一个 estimator $G_{\widehat{M}}^{\pi}(s,a)$
$$
G_{\widehat{M}}^{\pi}(s,a):=\sum_{s^{\prime}\sim\widehat{T}(s,a)}[V_{M}^{\pi}(s^{\prime})]-\sum_{s^{\prime}\sim T(s,a)}[V_{M}^{\pi}(s^{\prime})]
$$
$G_{\widehat{M}}^{\pi}(s,a)$ 将动力学模型的估计误差和奖励的估计误差联系了起来。它用 $V_\pi$ 衡量了$M$ 和 $\widehat{M}$ 之间的差异，当这两者相同时，$G_{\widehat{M}}^{\pi}(s,a)=0$。所以，也体体现了策略 π 在两个MDP中表现的差异。

measures the difference between M and Mc under the test function V π — indeed, if M = Mc, then Gπ Mc(s, a) = 0.





是真实误差的一个上界。

有公式 1：
$$
\eta_{\widehat M}(\pi)-\eta_M(\pi)=\gamma_{(s,a)\sim\rho_{\widehat T}^{\pi}}\left[G_{\widehat M}^{\pi}(s,a)\right]
$$
该式体现了策略 π 在两个MDP中表现的差异。如果能够获得RHS(右式，Right Hand Side)的上界或者预估右式，则可以用它作为 $\eta_M(\pi)$ 的估计误差的上限。

有推论 公式2：
$$
\eta_M(\pi)=\underset{(s,a)\sim\rho_{\hat{T}}^{\pi}}{\bar{\mathbb{E}}}\left[r(s,a)-\gamma G_{\widehat{M}}^{\pi}(s,a)\right]\geq\underset{(s,a)\sim\rho_{\hat{T}}^{\pi}}{\bar{\mathbb{E}}}\left[r(s,a)-\gamma|G_{\widehat{M}}^{\pi}(s,a)|\right]
$$
该式表明如果一个策略可以在 $\widehat M$ 中获得高回报同时最小化$G_{\widehat M}^{\pi}$ 将在真实的 MDP 中也获得高回报。







因为$V_{M}^{\pi}$ 不知道，计算$G_{\widehat M}^{\pi}$ 就很困难。考虑到这一点，我们就可以将$G_{\widehat M}^{\pi}$ 替换为 只依赖于动力学模型 $\widehat{T}$误差的上界。

设 $\mathcal{F}$ 为 从 $\mathcal{S}$ 映射到 $\mathbb{R}$ 的一个函数集合，其中包含了 $V_{M}^{\pi}$ 。

$d_\mathcal{F}$ 是 积分概率度量(integral probability metric, IPM)，

用于衡量两个概率分布 p(x) 和 q(x) 之间的“距离” (相似性)。

[积分概率度量](https://0809zheng.github.io/2022/12/06/ipm.html)



 IPMs are quite general and contain several other distance measures as special cases [61]. Depending on what we are willing to assume about V π M, there are multiple options to bound Gπ Mc by some notion of error of Tb, discussed in greater detail in Appendix A:













[利普希茨连续条件](https://0809zheng.github.io/2022/10/11/lipschitz.html)





[^安道龙的回答 - 知乎]:https://www.zhihu.com/question/53001128/answer/2715286504

[积分概率度量](https://0809zheng.github.io/2022/12/06/ipm.html)