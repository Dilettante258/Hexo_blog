---
title: 强化学习进阶-PPO 算法
categories: 强化学习
date: 2023-10-25 17:07:23
mathjax: true
---

参考：[Proximal Policy Optimization Algorithms - Arxiv](https://arxiv.org/abs/1707.06347)

## Abstract

OpenAI propose a new family of *policy gradient* methods for reinforcement learning, which <u>alternate between sampling data through interaction with the environment, and optimizing a “surrogate” objective function using stochastic gradient ascent</u>. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that <u>enables multiple epochs of minibatch updates</u>. The new methods, which we call ***<u>Proximal Policy Optimization (PPO)</u>***, have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). 

PPO (Proximal Policy Optimization) 是一种新型的强化学习策略梯度方法。它通过与环境交互来采样数据，并使用随机梯度上升来优化一个*“surrogate” objective function*。与标准的策略梯度方法不同，PPO提出了一个新的目标函数，允许进行多次的小批量更新。PPO结合了TRPO (Trust Region Policy Optimization) 的一些优点，但实现更简单（using only first-order optimization），适用性更广泛，且样本复杂性更低。实验结果显示，PPO在多个基准任务上，如模拟机器人行走和Atari游戏，均表现优于其他在线策略梯度方法，总体上在样本复杂性、简单性和实际运行时间上达到了一个很好的平衡。

## Background: Policy Optimization

### Policy Gradient Methods

Policy Gradient是DRL中一大类方法，核心思想就是直接优化策略网络Policy Network来提升Reward的获取。目标函数的梯度中有一项轨迹(trajectory)回报，用于指导策略的更新，可以把[梯度]写成下面这个一般的形式：
$$
g=\mathbb{E}\left[\sum_{t=0}^T\psi_t\nabla_\theta\log\pi_\theta(a_t|s_t)\right]
$$
其中$\psi_t$是对状态动作对的价值评估，可以有很多种形式。

$\psi_t$通常采用Advantage的方式更优，the most commonly used gradient estimator has the form
$$
\hat{g}=\hat{\mathbb{E}}_t\left[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\hat{A}_t\right]
$$

> Advantage可以认为是对动作a的相对评估。因为Policy会不断更新，好坏的标准是不断变的，Advantage意味着如果动作a在当前的policy下比标准好才是正的，差就是负的。但怎么准确的估计Advantage是个问题，核心是bias与variance的问题。如果我们使用真实的trajectory来估计，那么bias为0，但variance很大。如果我们使用one step value来估计，那么variance比较少，但bias很大。[^知乎]

### Trust Region Methods

#### $\psi_t$ of TRPO

TRPO（Trust Region Policy Optimization，信任区域策略优化）是基于基本的Policy Gradient进行改进的算法，TRPO方法不是使用所执行动作的对数概率梯度，而是使用了一个不同的目标：由优势值进行缩放的新策略和旧策略之间的比率。

TRPO的loss和前面的policy gradient并不一样，原因在于importance sampling的使用。

1. 由于On-Policy方法要求每次更新都使用新的按当前策略采样的数据，这可能会非常低效，尤其是当数据采样成本高或环境复杂时。
2. 虽然TRPO和PPO在理论上是On-Policy方法，但为了提高数据效率，它们实际上会多次使用同一批数据来进行多次更新。所以实际上大部分情况是off policy的，只能说是near on policy。
3. 一次采样的数据分minibatch训练神经网络迭代多次，一般我们还会重复利用一下数据，也就是sample reuse。
4. 由于在多次更新中重复使用数据，数据的分布可能与当前策略的分布不完全匹配。为了纠正这种不匹配，可以使用重要性采样（Importance Sampling）来调整。

在梯度计算上加上importance sampling: 
$$
\hat{g}=\hat{\mathbb{E}}_{t}\left[\left(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\mathrm{ld}}}(a_{t}|s_{t})}\frac{\pi_{\theta}(a_{t+1}|s_{t+1})}{\pi_{\theta_{\mathrm{old}}}(a_{t+1}|s_{t+1})}\ldots\frac{\pi_{\theta}(a_{T}|s_{T})}{\pi_{\theta_{\mathrm{ld}}}(a_{T}|s_{T})}\right)\nabla_{\theta}\log\pi_{\theta}\left(a_{t}\mid s_{t}\right)\hat{A}_{t}\right]
$$
取1步importance sampling ratio, 近似可得：
$$
\begin{aligned}
\hat{g}& \approx\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t}\mid s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t}\mid s_{t}\right)}\nabla_{\theta}\log\pi_{\theta}\left(a_{t}\mid s_{t}\right)\hat{A}_{t}\right]  \\
&=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t}\mid s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t}\mid s_{t}\right)}\frac{\nabla_{\theta}\pi_{\theta}\left(a_{t}\mid s_{t}\right)}{\pi_{\theta}\left(a_{t}\mid s_{t}\right)}\hat{A}_{t}\right] \\
&=\hat{\mathbb{E}}_t\left[\frac{\nabla_\theta\pi_\theta\left(a_t\mid s_t\right)}{\pi_{\theta_\mathrm{old}}\left(a_t\mid s_t\right)}\hat{A}_t\right]
\end{aligned}
$$
因此，有了优化目标：
$$
L^{PG}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t\mid s_t\right)}{\pi_{\theta_\mathrm{old}}\left(a_t\mid s_t\right)}\hat{A}_t\right]
$$
以上内容都是TRPO的Loss推导过程。

#### Kullback-Leibler Divergence

PG方法存在核心问题在于数据的bias。因为Advantage估计是不完全准确的，存在bias，那么如果Policy一次更新太远，那么下一次采样将完全偏离，导致Policy更新到完全偏离的位置，从而形成恶性循环。因此，TRPO的核心思想就是让每一次的Policy更新在一个Trust Region里，保证policy的单调提升。
$$
\begin{array}{rl}\text{maximize}&\hat{\mathbb{E}}_t\bigg[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\mathrm{old}}(a_t\mid s_t)}\hat{A}_t\bigg]\\\text{subject to}&\hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\mathrm{old}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]]\le\delta.\end{array}
$$
为了保证新旧策略足够接近（因为上面做了近似操作），TRPO 使用了**库尔贝克-莱布勒（Kullback-Leibler，KL）散度**来衡量策略之间的距离。这里的不等式（即第二行）约束定义了策略空间中的一个 KL 球，被称为信任区域。在这个区域中，可以认为当前学习策略和环境交互的状态分布与上一轮策略最后采样的状态分布一致，进而可以基于一步行动的重要性采样方法使当前学习策略稳定提升。

第一行的目标函数（“surrogate” objective）在策略更新大小的约束下被最大化。$\theta_{old}$是更新前的策略参数向量。在对目标进行线性近似和对约束进行二次近似后，可以使用共轭梯度算法有效地近似解决这个问题。

实际上，解释TRPO的理论建议使用惩罚而不是约束，即解决对于某个系数β无约束优化问题。这一现象是因为某个surrogate objective（它计算状态的最大KL而不是平均值）在策略π的性能上形成了一个下界（即，一个消极的界限）。TRPO使用硬约束而不是惩罚，因为很难选择一个 *在不同问题中都表现良好的β值* ——甚至在一个问题中，其中的特性在学习过程中会发生变化。因此，为了实现我们的目标，即模拟TRPO的单调改进的一阶算法，实验表明，仅仅选择一个固定的惩罚系数β并使用SGD优化被惩罚的<a href="#TRPO目标方程">目标方程</a>是不够的；还需要进行额外的修改。

<a name="TRPO目标方程">目标方程</a>：
$$
\operatorname*{maximize}_{\theta}\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\mid s_{t})}\hat{A}_{t}-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]\right]
$$
$\hat{A}_{t}$是优势函数，它表示在状态 $s_{t}$采取行动 $a_t$相对于平均的预期回报。

*β* 是一个惩罚系数，它控制策略变化的大小

> KL散度是衡量两个概率分布之间差异的一种方法。它经常在信息论、统计推断、机器学习等领域中被使用。
>
> 给定两个概率分布 P 和 Q，对于离散型随机变量，KL散度定义为：$D_{KL}(P||Q)=\sum_iP(i)\log\frac{P(i)}{Q(i)}$。对于连续型随机变量，KL散度定义为：$D_{KL}(P||Q)=\int P(x)\log\frac{P(t)}{Q(t)}dx$。其中，⁡log 是自然对数。
>
> 几点需要注意的是：
>
> 1.  KL散度是非负的，当且仅当 *P* 和 *Q* 完全相同时，KL散度为0。
> 2. KL散度不是对称的，也就是说$D_{KL}(P||Q)\neq D_{KL}(Q||P)$。
> 3. KL散度可以被理解为，当使用概率分布 *Q* 来近似 *P* 时，所损失的信息量。
>
> 在实际应用中，KL散度常常用于概率分布的拟合问题，例如在变分推断中，我们可能会选择一个简单的分布 *Q* 来近似复杂的后验分布 *P*，并通过最小化它们之间的KL散度来调整 *Q* 的参数。

### PPO

TRPO是loss的带条件的loss，用共轭梯度法进行计算，比较麻烦，不能大规模应用，因此需要做简化，能够直接使用梯度下降实现，因此PPO应运而生。

PPO的核心思想是限制策略更新的大小，以避免策略变化太大导致训练不稳定。PPO有两种主要的变体，一种是PPO-Clip，另一种是PPO-Penalty。这两种方法都试图限制策略更新的大小，但是方法有所不同。

#### Adaptive KL Penalty Coefficient

PPO-Penalty的核心思想是在目标函数中（用拉格朗日乘数法）加入一个惩罚项（对KL 散度的限制），这个惩罚项与策略的变化大小有关。这就变成了一个非硬约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数 β。

简单描述这一算法的过程：

- 使用几个epochs的小批量SGD，优化KL惩罚目标
  $$
  L^{KLPEN}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t-\beta\operatorname{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t),\pi_\theta(\cdot\mid s_t)]\right]
  $$

- 计算$d=\hat{\mathbb{E}}_{t}[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_{t}),\pi_{\theta}(\cdot\mid s_{t})]]$

  - If $d<d_{\mathrm{targ}}/1.5,\beta\leftarrow\beta/2$
  - If $d>d_{\mathrm{targ}}\times1.5,\beta\leftarrow\beta\times2$

更新后的β用于下一次策略更新。这种方案下，我们偶尔会看到KL散度与$d_{targ}$明显不同的策略更新，然而，这种情况并不常见，而且β会迅速调整。上面的参数1.5和2是试探性地选择的，但算法对它们不是很敏感。β的初始值是另一个超参数，但在实践中并不重要，因为算法会快速调整它。

PPO-Penalty的目标是最大化上述目标函数。当策略变化太大时，惩罚项会增加，从而限制策略的更新。

#### Clipped Surrogate Objective

PPO的主要贡献是提出了clipped surrogate objective，该形式 PPO-Clip 更加直接，它在目标函数中进行限制，以保证新的参数和旧的参数的差距不会太大。

算法在迭代更新时，观察当前策略在t 时刻智能体处于状态 s 所采取的行为概率${\pi_\theta(\alpha_t|s_t)}$，与之前策略所采取行为概率$\pi_{\theta old}(a_t|s_t)$，计算这些概率的比值来控制新策略更新幅度，比值 $r_t$ 记作：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta\:old}(a_t\:|s_t)}
$$
在TRPO算法中将最大化 “surrogate” objective
$$
L^{CPI}(\theta)=\hat{\mathbb{E}}_t\bigg[\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}\hat{A}_t\bigg]=\hat{\mathbb{E}}_t\bigg[r_t(\theta)\hat{A}_t\bigg]
$$
上标CPI 指的是保守策略迭代(conservative policy iteration)。 如果没有约束，最大化 $L^{CPI}$ 将导致过分的策略更新；因此，我们要考虑如何调整目标，来惩罚使策略从$r_t(\theta)$ 从1偏离的动作。

PPO算法中目标如下所示：
$$
L^{CLIP}(\theta)=\hat{\mathbb{E}}_t\Big[\min(r_t(\theta)\hat{A}_t,\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)\Big]
$$

其中 $clip( x, l, r) : = \max ( \min ( x, r) , l) $ ,把 $x$ 限制在 $[l,r]$ 内，上式表示限制比例$r_t$在$\left[1-\epsilon, 1+\epsilon\right]$。上式中$\epsilon$是一个超参数，一般**设定值为 0.2**，表示进行截断 (clip) 的范围。最终$L^{CLIP}(\theta)$借助 $min$ 函数选取未截断与截断目标之间的更小值，形成目标下限。

$L^{CLIP}(\theta)$**可以分为优势函数 A 为正数和负数两种情况**：

- 如果$A^{\pi e_k}(s,a)>0$，说明现阶段的$(s_t,a_t)$相对较好，那么我们希望该二元组出现的概率越高越好，即ratio中的分子越大越好，但是分母分子不能差太多，因此需要加一个上限，最大化这个式子会增大$r_t(\theta)$，但不会让其超过$1+\epsilon$。
- 如果$A^{\pi_{\theta_k}}(s,a)<0$，说明现阶段的$(s_t,a_t)$相对较差，那么我们希望该二元组出现的概率越低越好，即ratio中的分子越小越好，最大化这个式子会减小$r_t(\theta)$，但不会让其超过$1-\epsilon$。


PPO的核心思想很简单，对于ratio 也就是当前policy和旧policy的偏差做clip，如果ratio偏差超过 $\epsilon$ 就做clip，clip后**梯度也直接变为0，神经网络也就不再更新参数了**。这样，在实现上，无论更新多少步都没有关系，有clip给我们挡着，不担心训练偏了。[^深度解读：Policy Gradient，PPO及PPG-知乎]

#### GAE (Generalized Advantage Estimation)

回到[策略梯度](#Policy Gradient Methods)那里，$\psi_t$通常采用Advantage的方式更优，它的坏处就是不容易计算。 策略梯度使用全部奖励来估计策略梯度，尽管无偏但是方差大；Actor-Critic方法使用值函数来估计奖励，能够降低偏差但是方差较大。高方差需要更多的样本来训练，偏差会导致不收敛或收敛结果较差。[^GAE-CSDN]但为了更好的平衡偏差和方差的问题，一般采用GAE即$TD(\lambda)$的形式。

**Generalized Advantage Estimation (GAE)** 是强化学习中用于估计优势函数的方法。GAE的优点是它提供了一个平滑的方式来权衡偏差和方差之间的权衡。通过调整 *λ* 参数，可以在偏差和方差之间找到一个合适的平衡点。

为了理解GAE，首先要了解两个基本概念：TD误差 (Temporal Difference Error) 和优势函数 (Advantage Function)。

1. **TD误差**：TD误差是真实的回报与估计的回报之间的差异。在时间步 *t*，TD误差可以表示为：
   $$
   \delta_t^V=r_t+\lambda V(s_{t+1})-V(s_t)
   $$
   其中，$r_t$ 是在时间步$t$ 获得的奖励，$V(s)$ 是状态值函数，它估计了从状态 $s$ 开始并遵循当前策略的预期回报，而$\gamma$是折扣因子。

2. **优势函数**：优势函数度量了在状态 *s* 采取动作 *a* 相对于按照当前策略采取动作的期望值更好或更差多少。它可以表示为： 
   $$
   A(s,a)=Q(s,a)-V(s)
   $$
   其中，$Q(s,a)$是动作值函数，它估计了在状态 s 采取动作 a 并遵循当前策略的预期回报。所以只要我们能找到一个新策略，使得$A(s,a) > 0$就能保证策略性能单调递增。

GAE的核心思想是使用加权的TD误差来估计优势函数。具体来说，GAE定义了一个参数 *λ*，它在0和1之间，用于控制考虑的未来TD误差的数量和权重。具体推导过程很复杂，GAE只是用到了上面的两个函数作为基础，GAE的 advantage estimator为：

$$
A_{t}^{GAE(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V}
$$

拆开为：
$$
\hat{A}_t^{GAE(\gamma,\lambda)}=\delta_t^V+(\gamma\lambda)\delta_{t+1}^V+(\gamma\lambda)^2\delta_{t+2}^V+\cdots+(\gamma\lambda)^{T-t+1}\delta_{T+1}
$$
当 λ=0 的时候，GAE的形式就是TD误差的λ形式，有偏差，但方差小。 λ=1 的时候，就是蒙特卡洛的形式，无偏差，但是方差大。

具体实现的时候，不可能算无穷，就通过固定采样的长度n-step，采用递归的方式实现：
$$
A_t^{GAE(\gamma,\lambda)}=\sum_{l=0}^{T-t}((1-d_t)\gamma\lambda)^l\delta_{t+l}^V ,\quad T=t+n
$$

关于GAE具体推导过程，[GAE——泛化优势估计-知乎](https://zhuanlan.zhihu.com/p/356447099)(推荐）和[【DRL-13】Generalized Advantage Estimator-知乎](https://zhuanlan.zhihu.com/p/139097326)感觉讲的很清楚。

### Pseudocode

![image-20231028131324439](http://106.15.139.91:40027/uploads/2312/658d4e05be246.png)

| Version                        | Surrogate L                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| No clipping or penalty         | $L_{t}(\theta) =r_{t}(\theta)\hat{A}_{t}$                    |
| Clipping                       | $L_{t}(\theta) =\min(r_{t}(\theta)\hat{A}_{t},\operatorname{clip}(r_{t}(\theta)),1-\epsilon,1+\epsilon)\hat{A}_{t} $ |
| KL penalty (fixed or adaptive) | $L_{t}(\theta) =r_{t}(\theta)\hat{A}_{t}-\beta\:\mathrm{KL}[\pi_{\theta_{\mathrm{old}}},\pi_{\theta}] $ |

以上内容摘自论文。[^Proximal Policy Optimization Algorithms]

PPO 本质上基于 Actor-Critic 框架，算法流程如下：

![img](https://img-blog.csdnimg.cn/12115bc65807490e84166ef3440849bc.png)

PPO 算法主要由 Actor 和 Critic 两部分构成，Critic 部分更新方式与其他Actor-Critic 类型相似，通常采用计算 TD-error（时序差分误差）形式。对于 Actor 的更新方式，PPO 可在$L^{KLPEN}$ 、$L^{CLIP}$ 之间选择对于当前实验环境稳定性适用性更强的目标函数，经过 OpenAI 研究团队实验论证，PPO- Clip 比 PPO- Penalty有更好的数据效率和可行性。 [^PPO 模型解析-CSDN]





[^Proximal Policy Optimization Algorithms]:https://arxiv.org/abs/1707.06347
[^PPO 模型解析-CSDN]: https://blog.csdn.net/dgvv4/article/details/129496576
[^GAE-CSDN]:https://blog.csdn.net/weixin_39891381/article/details/105153867

[^深度解读：Policy Gradient，PPO及PPG-知乎]: https://zhuanlan.zhihu.com/p/342150033
