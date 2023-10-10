---
title: 张量系统（Tensor）和自动微分系统（autograd）
categories: PyTorch
date: 2023-09-14 16:00:00
---

Tensor和numpy的ndarrays类似，但PyTorch的tensor支持GPU加速。

从接口的角度讲，对tensor的操作可分为两类：

1. torch.function，如torch.save等。
2. tensor.function，如tensor.view等。

从存储的角度讲，对tensor的操作又可分为两类：

1. 不会修改自身的数据，如a.add(b)，加法的结果会返回一个新的tensor。
2. 会修改自身的数据，如a.add\_(b)，加法的结果仍存储在a中，a被修改了。函数名以\_结尾的都是inplace方式，即会修改调用者自己的数据，在实际应用中需加以区分。

常见的新建Tensor的方法

| 函数                              | 功能                        |
| --------------------------------- | --------------------------- |
| Tensor(*sizes)                    | 基础构造函数                |
| ones(*sizes)                      | 全1Tensor                   |
| zeros(*sizes)                     | 全0Tensor                   |
| eye(*sizes)                       | 对角线为1，其他为 0         |
| arange(s,e,step)                  | 从s到e，步长为step          |
| linspace(s,e,steps)               | 从s到e，均匀切分成 steps 份 |
| rand/randn(*sizes)                | 均匀/标准分布               |
| normal(mean,std)/uniform(from,to) | 正态分布/均匀分布           |
| randperm(m)                       | 随机排列                    |

`a = t.Tensor(2, 3)`数值取决于内存空间的状

`b = t.Tensor([[1,2,3],[4,5,6]])`用list的数据创建tensor