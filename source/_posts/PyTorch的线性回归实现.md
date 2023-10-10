---
title: PyTorch的线性回归实现
categories: PyTorch
date: 2023-09-16 15:25:00
mathjax: true
---

## 线性回归的从零开始实现

### 生成数据集

从零开始实现整个方法，包括数据流水线、模型、损失函数和小批量随机梯度下降优化器

```python
%matplotlib inline
import random
import torch
from d2l import torch as d2l
```

根据带有噪声的线性模型构造一个人造数据集。 我们使用线性模型参数$w=[2,−3.4]^⊤$、$b=4.2$和噪声项$ϵ$生成数据集及其标签：
$$
y=Xw+b+ϵ
$$

```python
def synthetic_data(w, b, num_examples):  
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

`features` 中的每一行都包含一个二维数据样本，`labels` 中的每一行都包含一维标签值（一个标量）

```python
print('features:', features[0], '\nlabel:', labels[0])
Out:
features: tensor([-0.6612, -1.8215]) 
label: tensor([9.0842])
```

```python
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
```

### 读取数据集

训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。

在下面的代码中，我们定义一个`data_iter`函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量。 每个小批量包含一组特征和标签。

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

使用例：

```python
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

### 初始化模型参数

在我们开始用小批量随机梯度下降优化我们的模型参数之前， 我们需要先有一些参数。 在下面的代码中，我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据。 每次更新都需要计算损失函数关于模型参数的梯度。 有了这个梯度，我们就可以向减小损失的方向更新每个参数。

### 定义模型

定义模型，将模型的输入和参数同模型的输出关联起来。 

广播机制： 当我们用一个向量加一个标量时，标量会被加到向量的每个分量上。

```python
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

### 定义损失函数

因为需要计算损失函数的梯度，所以我们应该先定义损失函数。 这里我们使用平方损失函数。 在实现中，我们需要将真实值`y`的形状转换为和预测值`y_hat`的形状相同。

```python
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 定义优化算法

这里使用小批量随机梯度下降。

在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。 该函数接受模型参数集合、学习速率和批量大小作为输入。每 一步更新的大小由学习速率`lr`决定。 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（`batch_size`） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。

```python
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

### 训练

在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。 计算完损失后，我们开始反向传播，存储每个参数的梯度。 最后，我们调用优化算法`sgd`来更新模型参数。

概括一下，我们将执行以下循环：

- 初始化参数
- 重复以下训练，直到完成
  - 计算梯度
  - 更新参数

在每个*迭代周期*（epoch）中，我们使用`data_iter`函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
```

## 线性回归的简洁实现

通过使用深度学习框架来简洁地实现线性回归模型

### 生成数据集

```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

### 读取数据集

我们可以调用框架中现有的API来读取数据。 我们将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。 此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。

```python
def load_array(data_arrays, batch_size, is_train=True):  
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter)) #next从迭代器中获取第一项。
```

### 定义模型

对于标准深度学习模型，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。 我们首先定义一个模型变量`net`，它是一个`Sequential`类的实例。 `Sequential`类将多个层串联在一起。 当给定输入数据时，`Sequential`实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。 在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。 

```python
# nn是神经网络的缩写
from torch import nn
#两个参数中第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。
net = nn.Sequential(nn.Linear(2, 1))
```

### 初始化模型参数

在使用`net`之前，我们需要初始化模型参数。 如在线性回归模型中的权重和偏置。 深度学习框架通常有预定义的方法来初始化参数。 在这里，我们指定<u>每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样， 偏置参数将初始化为零。</u>

正如我们在构造`nn.Linear`时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。 （`nn.Linear()`自带权重`weight`和偏置属性`bias`。）我们通过`net[0]`选择网络中的第一个图层， 然后使用`weight.data`和`bias.data`方法访问参数。 我们还可以使用替换方法`normal_`和`fill_`来重写参数值。

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

以下代码可以访问参数：

```python
for name, param in net.named_parameters():
    print(name, param.size(), type(param))
```

### 定义损失函数

计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数。 默认情况下，它返回所有样本损失的平均值。

```python
loss = nn.MSELoss()
```

### 定义优化算法

小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在`optim`模块中实现了该算法的许多变种。 当我们实例化一个`SGD`实例时，我们要指定优化的参数 （可通过`net.parameters()`从我们的模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

### 训练

通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码。 我们不必单独分配参数、不必定义我们的损失函数，也不必手动实现小批量随机梯度下降。 当我们需要更复杂的模型时，高级API的优势将大大增加。 当我们有了所有的基本组件，训练过程代码与我们从零开始实现时所做的非常相似。

回顾一下：在每个迭代周期里，我们将完整遍历一次数据集（`train_data`）， 不停地从中获取一个小批量的输入和相应的标签。 对于每一个小批量，我们会进行以下步骤:

- 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
- 通过进行反向传播来计算梯度。
- 通过调用优化器来更新模型参数。

为了更好的衡量训练效果，我们计算每个迭代周期后的损失，并打印它来监控训练过程。

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
```