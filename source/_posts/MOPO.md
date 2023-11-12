---
title: 离线强化学习-MOPO
categories: 强化学习 离线强化学习
date: 2023-11-12 12:30:23
mathjax: true
---



目前主流的offline RL的方法都是model free的，这类方法通常需要将policy限制到data覆盖的集合范围里（support），不能泛化到没见过的状态上。作者提出Model-based Offline Policy Optimization (MOPO)算法,用model based的方法来做offline RL，同时通过给reward添加惩罚项（soft reward penalty）来描述环境转移的不确定性（applying them with rewards artificially penalized by the uncertainty of the dynamics.）这种方式相当于在泛化性和风险之间做tradeoff。作者的意思是，这种方式允许算法为了更好的泛化性而承担一定风险（policy is allowed to take a few risky actions and then return to the confident area near the behavioral distribution without being terminated）。具体做法就是，先根据data去学一堆状态转移函数，这个函数是一个用神经网络表示的关于状态和reward的高斯分布

MOPO: Model-based Offline Policy Optimization

离线强化学习（Offline reinforcement learning, RL）指的是完全从一批先前收集的数据中学习策略的问题。尽管无模型离线RL取得了显著进展，但最成功的现有方法将策略限制在数据支持范围内，无法推广到新状态。
在本文中，作者观察到，与无模型方法相比，现有的基于模型的RL算法本身已经在离线设置中产生了显著的优势，尽管它并不是为这种设置而设计的。然而，尽管许多基于标准模型的RL方法已经估计了模型的不确定性，但它们本身并没有提供避免与离线环境中的分布偏移相关的问题的机制。因此，作者建议修改现有的基于模型的RL方法，以通过将基于模型的脱机RL转换为受惩罚的MDP框架来解决这些问题。
作者从理论上表明，通过使用这种惩罚性MDP，可以使真实MDP中的收益下限最大化。基于上述理论结果，作者提出了一种新的基于模型的离线RL算法，该算法将Lipschitz正则化模型的方差作为对奖励函数的惩罚。实验表明，该算法的性能优于现有的离线RL基准测试中基于标准模型的RL方法和现有的最新无模型离线RL方法，以及两项具有挑战性的需要概括从不同任务收集的数据连续控制任务。



MOPO是一种model-based offline RL方法，简单来说就是把MBPO用在了offline设定下，根据offline的需要做了一些小修改。

一方面，这种model-based方法生成 D_model 数据集，能够提升sample efficiency；另一方面，MOPO能够利用model让策略以一定概率选择行为分布之外的action。但是离行为分布越远，不确定性会越大，所以MOPO通过不确定性估计来量化偏离行为分布的风险，实现其和探索多样state所得收益之间的trade-off。具体来说，就是在reward函数后面加了一个不确定性惩罚项，通过ensemble的方法（即N个dynamics模型的最大标准差）量化这种不确定性，不确定性越大则reward越小。也就是说不是直接惩罚OOD状态和动作，而是惩罚具有高不确定性的 (s,a) 。