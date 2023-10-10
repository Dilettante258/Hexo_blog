---
title: softmax回归基础实践
categories: PyTorch
date: 2023-09-18 16:08:00
mathjax: true
---

# 准备阶段-图像分类数据集

使用Fashion-MNIST数据集。

```python
%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
```

## 读取数据集

我们可以通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。

```python
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

len(mnist_train), len(mnist_test)
Out:(60000, 10000)
```

Fashion-MNIST由10个类别的图像组成， 每个类别由*训练数据集*（train dataset）中的6000张图像 和*测试数据集*（test dataset）中的1000张图像组成。 因此，训练集和测试集分别包含60000和10000张图像。 测试数据集不会用于训练，只用于评估模型性能。

```python
mnist_train[0][0].shape
Out:torch.Size([1, 28, 28])
```

每个输入图像的高度和宽度均为28像素。 数据集由灰度图像组成，其通道数为1。 为了简洁起见，本书将高度ℎ像素、宽度w像素图像的形状记为ℎ×w或（ℎ,w）。

```python
def get_fashion_mnist_labels(labels):  
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

第一个函数用于在数字标签索引及其文本名称之间进行转换。第二个函数可视化这些样本。

运用：

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

![../_images/output_image-classification-dataset_e45669_83_0.svg](https://zh-v2.d2l.ai/_images/output_image-classification-dataset_e45669_83_0.svg)

## 读取小批量

为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。 回顾一下，在每次迭代中，数据加载器每次都会读取一小批量数据，大小为`batch_size`。 通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。

```python
batch_size = 256

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## 整合所有组件

现在我们定义`load_data_fashion_mnist`函数，用于获取和读取Fashion-MNIST数据集。 这个函数返回训练集和验证集的数据迭代器。 此外，这个函数还接受一个可选参数`resize`，用来将图像大小调整为另一种形状。

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

下面，我们通过指定`resize`参数来测试`load_data_fashion_mnist`函数的图像大小调整功能。

```python
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

# 从零开始实现

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256 #设置数据迭代器的批量大小为256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

和之前线性回归的例子一样，这里的每个样本都将用固定长度的向量表示。 原始数据集中的每个样本都是28×28的图像。 本节将展平每个图像，把它们看作长度为784的向量。 在后面的章节中，我们将讨论能够利用图像空间结构的特征， 但现在我们暂时只把每个像素位置看作一个特征。

回想一下，在softmax回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个784×10的矩阵， 偏置将构成一个1×10的行向量。 与线性回归一样，我们将使用正态分布初始化我们的权重W，偏置初始化为0。

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

## 定义softmax操作

在实现softmax回归模型之前，我们简要回顾一下`sum`运算符如何沿着张量中的特定维度工作。 给定一个矩阵`X`，我们可以对所有元素求和（默认情况下）。 也可以只求同一个轴上的元素，即同一列（轴0）或同一行（轴1）。 如果`X`是一个形状为`(2, 3)`的张量，我们对列进行求和， 则结果将是一个具有形状`(3,)`的向量。 当调用`sum`运算符时，我们可以指定保持在原始张量的轴数，而不折叠求和的维度。 这将产生一个具有形状`(1, 3)`的二维张量。

```python
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)
Out:
(tensor([[5., 7., 9.]]),
 tensor([[ 6.],
         [15.]]))
```

而实现softmax由三个步骤组成：

1. 对每个项求幂（使用`exp`）；
2. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
3. 将每一行除以其规范化常数，确保结果的和为1。(分母或规范化常数，有时也称为*配分函数*（其对数称为对数-配分函数）。 该名称来自[统计物理学](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))中一个模拟粒子群分布的方程。)

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

上述代码，对于任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1。

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
Out:
(tensor([[0.1686, 0.4055, 0.0849, 0.1064, 0.2347],
         [0.0217, 0.2652, 0.6354, 0.0457, 0.0321]]),
 tensor([1.0000, 1.0000]))
```

注意：矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，但我们没有采取措施来防止这点。

## 定义模型

定义softmax操作后，我们可以实现softmax回归模型。 下面的代码定义了输入如何通过网络映射到输出。 注意，将数据传递到模型之前，我们使用`reshape`函数将每张原始图像展平为向量。

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

## 定义损失函数

接下来实现交叉熵损失函数。 这可能是深度学习中最常见的损失函数，因为目前分类问题的数量远远超过回归问题的数量。

回顾一下，交叉熵采用真实标签的预测概率的负对数似然。 这里我们不使用Python的for循环迭代预测（这往往是低效的）， 而是通过一个运算符选择所有元素。 下面，我们创建一个数据样本`y_hat`，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签`y`。 有了`y`，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 然后使用`y`作为`y_hat`中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。

```python
y = torch.tensor([0, 2]) #正确标签的索引
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
Out:	tensor([2.3026, 0.6931])
```

## 分类精度

给定预测概率分布`y_hat`，当我们必须输出硬预测（hard prediction）时， 我们通常选择预测概率最高的类。 许多应用都要求我们做出选择。如Gmail必须将电子邮件分类为“Primary（主要邮件）”、 “Social（社交邮件）”“Updates（更新邮件）”或“Forums（论坛邮件）”。 Gmail做分类时可能在内部估计概率，但最终它必须在类中选择一个。

当预测与标签分类`y`一致时，即是正确的。 分类精度即正确预测数量与总预测数量之比。 虽然直接优化精度可能很困难（因为精度的计算不可导）， 但精度通常是我们最关心的性能衡量标准，我们在训练分类器时几乎总会关注它。

为了计算精度，我们执行以下操作。 首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用`argmax`获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实`y`元素进行比较。 由于等式运算符“`==`”对数据类型很敏感， 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量。

```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
Out: 0.5
```

我们将继续使用之前定义的变量`y_hat`和`y`分别作为预测的概率分布和标签。 可以看到，第一个样本的预测类别是2（该行的最大元素为0.6，索引为2），这与实际标签0不一致。 第二个样本的预测类别是2（该行的最大元素为0.5，索引为2），这与实际标签2一致。 因此，这两个样本的分类精度率为0.5。

同样，对于任意数据迭代器`data_iter`可访问的数据集， 我们可以评估在任意模型`net`的精度。`numel`函数获取tensor中一共包含多少个元素。

```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        # 代码通过判断net是否为torch.nn.Module的实例来确定模型的类型
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 代码创建了一个Accumulator对象metric，用于累加正确预测数和预测总数
    with torch.no_grad(): # 使用上下文管理器来禁用梯度计算，以减少内存消耗。
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
            # 代码通过遍历数据迭代器data_iter，对每个输入X和标签y计算模型的预测结果，并使用metric.add()方法累加正确预测数和预测总数
    return metric[0] / metric[1] # 返回精度值
```

这里定义一个实用程序类`Accumulator`，用于对多个变量进行累加。 在上面的`evaluate_accuracy`函数中， 我们在`Accumulator`实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 当我们遍历数据集时，两者都将随着时间的推移而累加。

```python
class Accumulator:  #@save 
    # 定义了一个Accumulator类
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n # 初始化Accumulator对象，创建一个长度为n的列表self.data，并将列表中的每个元素初始化为0.0。
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 将传入的参数与self.data中的对应元素相加，并将结果保存回self.data中。参数args是一个可变参数，可以传入任意个数的参数。
    def reset(self):
        self.data = [0.0] * len(self.data) # 将self.data中的每个元素重置为0.0。

    def __getitem__(self, idx):
        return self.data[idx] # 返回self.data中索引为idx的元素。
    
evaluate_accuracy(net, test_iter)
Out: 0.0625
```

## 训练

 首先，我们定义一个函数来训练一个迭代周期。 请注意，`updater`是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是`d2l.sgd`函数，也可以是框架的内置优化函数。

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train() # 将模型设置为训练模式
    metric = Accumulator(3) # 记录训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X) #模型net() 函数将输入数据X传递给模型net，得到预测结果y_hat
        l = loss(y_hat, y) # 将预测结果y_hat与真实标签y传递给损失函数loss计算得到损失值l
        if isinstance(updater, torch.optim.Optimizer): # 如果优化器是.Optimizer的实例
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad() # 梯度清零
            l.mean().backward() # 计算梯度
            updater.step() 		# 计算梯度
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()	# 计算梯度
            updater(X.shape[0])	# 计算梯度
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel()) #更新记录
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
	# 将训练损失总和除以样本数，得到平均训练损失，并将训练准确度总和除以样本数，得到平均训练准确度。
```

在展示训练函数的实现之前，我们定义一个在动画中绘制数据的实用程序类`Animator`， 它能够简化本书其余部分的代码。

```python
class Animator:  #@save
    """在动画中绘制数据"""
    # 初始化Animator对象，设置图表的参数，如x轴标签，y轴标签，图例，坐标轴范围等。
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
	# 向图表中添加数据点，可以一次添加多个数据点。如果数据点的数量与之前添加的数据点数量不一致，则会创建新的数据列表。然后根据给定的格式绘制数据点，并更新图表。
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

接下来我们实现一个训练函数， 它会在`train_iter`访问到的训练数据集上训练一个模型`net`。 该训练函数将会运行多个迭代周期（由`num_epochs`指定）。 在每个迭代周期结束时，利用`test_iter`访问到的测试数据集对模型进行评估。 我们将利用`Animator`类来可视化训练进度。

这是一个名为`train_ch3`的<a name="train_ch3">函数</a>，用于训练模型。它接受以下参数：

- `net`：要训练的模型。
- `train_iter`：训练数据集的迭代器。
- `test_iter`：测试数据集的迭代器。
- `loss`：损失函数。
- `num_epochs`：训练的总轮数。
- `updater`：更新模型参数的函数。

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    # 在训练过程中绘制训练损失、训练准确率和测试准确率的曲线图。
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) # 对模型进行训练
        test_acc = evaluate_accuracy(net, test_iter) # 计算测试准确率
        animator.add(epoch + 1, train_metrics + (test_acc,)) # 将当前epoch的训练损失、训练准确率和测试准确率添加到曲线图中
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    # 如果有任何一个断言失败，将会引发一个AssertionError。
```

作为一个从零开始的实现，我们使用*小批量随机梯度下降*来优化模型的损失函数，设置学习率为0.1。

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

现在，我们训练模型10个迭代周期。 请注意，迭代周期（`num_epochs`）和学习率（`lr`）都是可调节的超参数。 通过更改它们的值，我们可以提高模型的分类精度。

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

![../_images/output_softmax-regression-scratch_a48321_222_0.svg](https://zh-v2.d2l.ai/_images/output_softmax-regression-scratch_a48321_222_0.svg)

## 预测

现在训练已经完成，我们的模型已经准备好对图像进行分类预测。 给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）。

```python
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmaxargmax(axis=1)) #argmax选择概率最大的
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

![../_images/output_softmax-regression-scratch_a48321_237_0.svg](https://zh-v2.d2l.ai/_images/output_softmax-regression-scratch_a48321_237_0.svg)

# 简洁实现

深度学习框架的高级API能够使实现线性回归变得更加容易。

```py
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

softmax回归的输出层是一个全连接层。 因此，为了实现我们的模型， 我们只需在`Sequential`中添加一个带有10个输出的全连接层。 同样，在这里`Sequential`并不是必要的， 但它是实现深度模型的基础。 我们仍然以均值0和标准差0.01随机初始化权重。

```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

## Softmax的实现

数学上的探讨：[交叉熵损失指数数值稳定性问题](https://zh-v2.d2l.ai/chapter_linear-networks/softmax-regression-concise.html#subsec-softmax-implementation-revisited)
下面的代码中参数reduction='none'表示不对损失进行求和或平均，而是返回每个样本的损失值。每个训练周期结束后对每个样本的损失进行个别处理，然后使用这些值进行可视化或其他分析。

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

## 优化算法

在这里，我们使用学习率为0.1的小批量随机梯度下降作为优化算法。 这与我们在线性回归例子中的相同，这说明了优化器的普适性。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

## 训练

调用之前定义的<a href="#train_ch3">训练函数</a>来训练模型。

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

![../_images/output_softmax-regression-concise_75d138_66_0.svg](https://zh-v2.d2l.ai/_images/output_softmax-regression-concise_75d138_66_0.svg)

结果收敛到一个相当高的精度，而且这次的代码比之前更精简了。



