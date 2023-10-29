---
title: 强化学习进阶-SAC 算法
categories: 强化学习
date: 2023-10-28 15:15:23
mathjax: true
---

无模型的深度强化学习算法已经在一系列的决策和控制任务上得到了证明。但是，这些方法通常面临两大挑战：非常高的样本复杂性和脆弱的收敛特性，这需要精心的超参数调整。这两个挑战严重限制了这些方法在复杂、真实世界领域的应用性。（训练非常不稳定，收敛性较差，对超参数比较敏感，也难以适应不同的复杂环境。）

在本文中，我们提出了基于最大熵强化学习框架的Soft Actor-Critic（SAC）的离线策略算法。在这个框架中，Actor 旨在最大化预期奖励，同时也最大化熵。也就是说，在尽可能随机的行动下成功完成任务。SAC 的前身是 Soft Q-learning，它们都属于最大熵强化学习的范畴。Soft Q-learning 不存在一个显式的策略函数，而是使用一个函数Q的波尔兹曼分布，在连续空间下求解非常麻烦。目前，在无模型的强化学习算法中，SAC 是一个非常高效的算法，它学习一个随机性策略，在不少标准环境中取得了领先的成绩。



## 最大熵强化学习

SAC的一个核心特征是熵正则化。策略被训练成在预期回报和熵之间找到一个权衡，其中熵是策略中的随机性度量。这与exploration-exploitation的权衡有密切关系：增加熵会导致更多的exploration，从而可以加速后续的学习。它还可以防止策略过早地收敛到不良的局部最优解。

**熵**（entropy）表示对一个随机变量的随机程度的度量。具体而言，如果 X 是一个随机变量，且它的概率密度函数为 p，那么它的熵 H 就被定义为
$$
H(X)=\mathbb{E}_{x\sim p}[-\log p(x)]
$$
在强化学习中，我们可以使用$H(\pi(\cdot|s))$来表示策略 π 在状态 s 下的随机程度。

**最大熵强化学习**（maximum entropy RL）的思想就是除了要最大化累积奖励，还要使得策略更加随机。如此，强化学习的目标中就加入了一项熵的正则项，定义为
$$
\pi^*=\arg\max_\pi\mathbb{E}_\pi\left[\sum_tr(s_t,a_t)+\alpha H(\pi(\cdot|s_t))\right]
$$
其中，α 是一个正则化的系数，用来控制熵的重要程度。

熵正则化增加了强化学习算法的探索程度，α 越大，探索性就越强，有助于加速后续的策略学习，并减少策略陷入较差的局部最优的可能性。

## Soft 策略迭代

在最大熵强化学习框架中，由于目标函数发生了变化，其他的一些定义也有相应的变化。

首先，我们看一下 Soft Bellmen equation：
$$
Q(s_t,a_t)=r(s_t,a_t)+\gamma\mathbb{E}_{s_{t+1}}[V(s_{t+1})]
$$
其中，状态价值函数被写为：
$$
V(s_t)=\mathbb{E}_{a_t\sim\pi}[Q(s_t,a_t)-\alpha\log\pi(a_t|s_t)]=\mathbb{E}_{a_t\sim\pi}[Q(s_t,a_t)]+H(\pi(\cdot|s_t))
$$
于是，根据该 Soft 贝尔曼方程，在有限的状态和动作空间情况下，Soft 策略评估可以收敛到策略 π 的 Soft  Q函数。然后，根据如下 Soft 策略提升公式可以改进策略：
$$
\pi_{\mathrm{new}}=\arg\min_{\pi^{\prime}}D_{KL}\left(\pi^{\prime}(\cdot|s),\frac{\exp(\frac1\alpha Q^{\pi_{\mathrm{old}}}(s,\cdot))}{Z^{\pi_{\mathrm{old}}}(s,\cdot)}\right)
$$

## SAC

在 SAC 算法中，我们为两个动作价值函数$Q$ (参数分别为$\omega_{1}$和$\omega_{2})$ 和一个策略函数$\pi$ (参数为$\theta$)建模。基于 Double DQN 的思想，SAC 使用两个$Q$网络，但每次用$Q$网络时会挑选一个$Q$值小的网络，从而缓解$Q$值过高估计的问题。SAC使用一个最小化Q值的目标Q网络来生成一个保守的、悲观估计，以确保 agent 不会高估Q值。这在训练中非常有用，因为高估Q值可能导致代理做出不稳定的、不可靠的决策，甚至可能导致不良策略。这有助于提高训练的稳定性和鲁棒性，使得代理更容易学习到高质量的策略。

任意一个函数$Q$的损失函数为：
$$
\begin{aligned}
L_{Q}(\omega)& =\mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim R}\left[\frac{1}{2}\left(Q_\omega(s_t,a_t)-(r_t+\gamma V_{\omega^-}(s_{t+1}))\right)^2\right]  \\
&=\mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim R,a_{t+1}\sim\pi_t(\cdot|s_{t+1})}\left[\frac{1}{2}\left(Q_\omega(s_t,a_t)-(r_t+\gamma(\min_{j=1,2}Q_{\omega_j}(s_{t+1},a_{t+1})-\alpha\log\pi(a_{t+1}|s_{t+1})))\right)^2\right]
\end{aligned}
$$
其中，$R$是策略过去收集的数据，因为 SAC 是一种离线策略算法。为了让训练更加稳定，这里使用了目标$Q$网络$Q_{\omega^-}$,同样是两个目标$Q$网络，与两个$Q$网络一一对应。SAC 中目标$Q$网络的更新方式与DDPG 中的更新方式一样。

策略$\pi$的损失函数由 KL 散度得到，化简后为：
$$
L_\pi(\theta)=\mathbb{E}_{s_t\sim R,a_t\sim\pi_\theta}[\alpha\log(\pi_\theta(a_t|s_t))-Q_\omega(s_t,a_t)]
$$
可以理解为最大化函数$V$,因为有$V(s_t)=\mathbb{E}_{a_t\sim\pi}[Q(s_t,a_t)-\alpha\log\pi(a_t|s_t)]$。对连续动作空间的环境，SAC 算法的策略输出高斯分布的均值和标准差，但是根据高斯分布来采样动作的过程是不可导的。因此，我们需要用到重参数化技巧 (reparameterization trick)。 **重参数化的做法是先从一个单位高斯分布$N$采样，再把采样值乘以标准差后加上均值。这样就可以认为是从策略高斯分布采样，并且这样对于策略函数是可导的。**我们将其表示为$a_t=f_\theta(\epsilon_t;s_t)$,其中$\epsilon_t$是一个噪声随机变量。同时考虑到两个函数$Q$,重写策略的损失函数：
$$
L_\pi(\theta)=\mathbb{E}_{s_t\sim R,\epsilon_t\sim\mathcal{N}}\left[\alpha\log(\pi_\theta(f_\theta(\epsilon_t;s_t)|s_t))-\min_{j=1,2}Q_{\omega_j}(s_t,f_\theta(\epsilon_t;s_t))\right]
$$
在 SAC 算法中，如何选择熵正则项的系数非常重要。在不同的状态下需要不同大小的熵： 在最优动作不确定的某个状态下，熵的取值应该大一点； 而在某个最优动作比较确定的状态下，熵的取值可以小一点。为了自动调整熵正则项，SAC 将强化学习的目标改写为一个带约束的优化问题：

$$
\max _\pi\mathbb{E} _\pi\left [ \sum _tr( s_t, a_t) \right ] \quad s.t.\quad \mathbb{E} _{( s_t, a_t) \sim \rho_\tau}[ - \log ( \pi_t( a_t|s_t) ) ] \geq \mathcal{H} _0
$$
也就是最大化期望回报，同时约束熵的均值大于$\mathcal{H}_{0}$。通过一些数学技巧化简后，得到α的损失函数：
$$
L(\alpha)=\mathbb{E}_{s_t\sim R,a_t\sim\pi(\cdot|s_t)}[-\alpha\log\pi(a_t|s_t)-\alpha\mathcal{H}_0]
$$
即当策略的熵低于目标值$\mathcal{H}_0$时，训练目标$L(\alpha)$会使$\alpha$的值增大，进而在上述最小化损失函数$L_{\pi}(\theta)$ 的过程中增加了策略熵对应项的重要性； 而当策略的熵高于目标值$\mathcal{H}_0$时，训练目标$L(\alpha)$会使$\alpha$的值减小，进而使得策略训练时更专注于价值提升。

算法流程：

![image-20231029084812263](images/image-20231029084812263.png)



![img](https://img-blog.csdnimg.cn/20190617142028132.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RheWRheWp1bXA=,size_16,color_FFFFFF,t_70)
