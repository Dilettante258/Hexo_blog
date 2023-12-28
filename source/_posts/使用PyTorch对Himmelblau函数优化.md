---
title: 使用PyTorch对Himmelblau函数优化
categories: PyTorch
date: 2023-09-16 14:25:00
---

Himmelblau函数是一个自变量大小为(2,)的4次函数，它得名于数学家David Himmelblau。该函数常用于测试优化器的性能。

```python
     def himmelblau(x):
         return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
```



## 绘制Himmelblau函数图像

```python
     %matplotlib inline
     import numpy as np
     from mpl_toolkits.mplot3d import Axes3D
     import matplotlib.pyplot as plt
     from matplotlib import cm
     from matplotlib.colors import LinearSegmentedColormap
     
     x = np.arange(-6, 6, 0.1)
     y = np.arange(-6, 6, 0.1)
     X, Y = np.meshgrid(x, y)
     Z = himmelblau([X, Y])
     
     fig = plt.figure()
     ax = fig.gca(projection='3d')
     ax.plot_surface(X, Y, Z)
     ax.view_init(60, -30)
     ax.set_xlabel('x[0]')
     ax.set_ylabel('x[1]')
     fig.show();
```

- **导入必要的包 。**注意，语句“`from mpl_toolkits.mplot3d import Axes3D`”是必须的，而且必须在导入matplotlib之前运行。虽然在后面的代码中并没有用到Axes3D，但是只有导入Axes3D之后再导入matplotlib，才能支持三维图表的绘制。
- **准备图表数据 。**这部分代码借助了扩展包numpy。这部分代码先用`np.arange()`函数分别准备X轴的采样点（一维）和Y轴的采样点（一维），再调用`np.meshgrid()`函数将X轴的采样点和Y轴的采样点整合起来，得到二维的采样点。这些二维的采样点表示了整个自变量空间。接着，调用在代码清单4-3中定义的函数`himmelblau()`，得到自变量空间上每个样点对应的函数值。
- **绘制图表 。**要绘制图4-4所示的表面图（surface plot），可以使用Axes3D类实例的plot_surface()函数。Axes3D类就是在导入部分导入的那个类。由代码“`ax=fig.gca(projection='3d')`”获得了Axes3D的类实例ax。接着调用这个类实例ax的成员方法plot_surface()绘制表面图。这个函数的3个参数分别代表每个样点的X轴坐标、Y轴坐标和Z轴坐标。接着，还要指定用什么视角来观察这个三维图。三维图的视角可以由仰角（elevation angle）和方位角（azimuth angle）这两个角完全确定。Axes3D类的`view_init()`函数的参数就是仰角和方位角的角度值（而不是弧度值）。

## 求解Himmelblau的最小值

要用torch.optim子包提供的优化器进行优化，需要完成以下步骤。

- 在构造torch.Tensor类实例时，用关键字参数`requires_grad`告诉PyTorch哪个张量是求梯度时需要考虑的自变量。这时还需要设置自变量的初始值。
- 选择优化器，并且告诉优化器哪些是要优化的决策变量。一般情况下，要优化的决策变量就是哪些求梯度时的自变量。还需要设置优化器的参数。
- 迭代优化。在初始化完成后，使用循环，反复求函数、调用优化器的成员方法step()改变的自变量的值。

```python
import torch
x = torch.tensor([0., 0.], requires_grad=True)
optimizer = torch.optim.Adam([x,])
for step in range(20001):
    if step:
        optimizer.zero_grad()
    f.backward()
    optimizer.step()
    f = himmelblau(x)
    if step % 1000 == 0:
        print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))
```

这段代码是使用PyTorch实现了对Himmelblau函数的优化。Himmelblau函数是一个有四个局部最小值的函数，用于测试优化算法的性能。

代码中首先导入了torch库，然后创建了一个包含两个元素的张量x，并将其设置为需要求导。接下来，创建了一个Adam优化器，将x作为要优化的参数传递给优化器。

然后，通过一个循环进行优化。在每个步骤中，首先调用`optimizer.zero_grad()`将梯度置零，然后计算Himmelblau函数的值f，并保存在变量f中。如果步骤不是第一步，就调用`f.backward()`计算梯度，并调用`optimizer.step()`根据计算的梯度更新参数x。

在每1000个步骤中，打印当前步骤数、参数x的值和对应的函数值f。

整个循环的目标是通过优化器来最小化Himmelblau函数，以找到最优的参数x，使得函数f的值最小。

## 求解Himmelblau的局部极大值

Himmelblau函数有4个最小值点，但是没有最大值点。这是因为Himmelblau函数可以任意大。但是Himmelblau函数在(-0.27, -0.92)附近有一个局部极大值。要用解析的方法找到这个值就不那么容易了。

由于torch.optim子包里的优化器都试图最小化函数的值，所以要对函数值取负号。对函数值取负号后求梯度得到的梯度值，与不取负号得到的梯度值正好相反。利用取反后的梯度值更新自变量，就可以最大化函数。对于Himmelblau函数的例子，只需要把代码中的`f.backward()`改为`(-f).backward()`。

其他代码不变（包括初始值和优化器等都不变）。运行更改过的代码，就可以找到局部极大值，其对应的函数值约为181.6。