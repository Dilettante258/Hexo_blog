---
title: IPython和Jupyter Notebooks
categories: Python
date: 2023-8-6 9:40:00
---

### Tab补全

在shell中输入表达式，按下Tab，会搜索已输入变量（对象、函数等等）的命名空间

```python
In [1]: an_apple = 27
In [2]: an_example = 42
In [3]: an<Tab>
an_apple    and         an_example  any
```

也可以补全任何对象的方法和属性：

```python
In [3]: b = [1, 2, 3]
In [4]: b.<Tab>
b.append  b.count   b.insert  b.reverse
b.clear   b.extend  b.pop     b.sort
b.copy    b.index   b.remove
```

也适用于模块：

```python
In [1]: import datetime

In [2]: datetime.<Tab>
datetime.date          datetime.MAXYEAR       datetime.timedelta
datetime.datetime      datetime.MINYEAR       datetime.timezone
datetime.datetime_CAPI datetime.time          datetime.tzinfo
```

当输入看似文件路径时（即使是Python字符串），按下Tab也可以补全电脑上对应的文件信息：

```python
In [7]: path = 'datasets/movielens/<Tab>
datasets/movielens/movies.dat    datasets/movielens/README
datasets/movielens/ratings.dat   datasets/movielens/users.dat
```

### 自省

在变量前后使用问号`?`，可以显示对象的信息。

如果对象是一个函数或实例方法，定义过的文档字符串，也会显示出信息。使用??会显示函数的源码。

`?`还有一个用途，就是像Unix或Windows命令行一样搜索IPython的命名空间。字符与通配符结合可以匹配所有的名字。例如，我们可以获得所有包含load的顶级NumPy命名空间：

```python
In [13]: np.*load*?
np.__loader__
np.load
np.loads
np.loadtxt
np.pkgload
```

### `%run`命令

你可以用`%run`命令运行所有的Python程序。

```python
In [14]: %run ipython_script_test.py
```

在Jupyter notebook中，你也可以使用`%load`，它将脚本导入到一个代码格

```python
>>> %load ipython_script_test.py

    def f(x, y, z):
        return (x + y) / z
    a = 5
    b = 6
    c = 7.5

    result = f(a, b, c)
```

### 中断运行的代码

代码运行时按Ctrl-C，都会导致`KeyboardInterrupt`。这会导致几乎所有Python程序立即停止，除非一些特殊情况。

### 从剪贴板执行程序

如果使用Jupyter notebook，你可以将代码复制粘贴到任意代码格执行。在IPython shell中也可以从剪贴板执行。

最简单的方法是使用`%paste`和`%cpaste`函数。`%paste`可以直接运行剪贴板中的代码.

`%cpaste`功能类似，但会给出一条提示：

```python
In [18]: %cpaste
Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
:x = 5
:y = 7
......
```

使用`%cpaste`，你可以粘贴任意多的代码再运行。你可能想在运行前，先看看代码。如果粘贴了错误的代码，可以用Ctrl-C中断。

### 键盘快捷键

Ipython快捷键：

| 快捷键          | 说明                                |
| --------------- | ----------------------------------- |
| Ctrl-P 或↑箭头 | 用当前输入的文本搜索之前的命令      |
| Ctrl-N或↓箭头    | 用当前输入的文本搜索之后的命令      |
| CtrI-R          | Readline 方式翻转历史搜索(部分匹配) |
| Ctrl-shift-V    | 从剪贴板粘贴文本                    |
| CtrI-C          | 中断运行的代码                      |
| Ctrl-A          | 将光标移动到一行的开头              |
| Ctrl-E          | 将光标移动到一行的末尾              |
| Ctrl-K          | 删除光标到行尾的文本                |
| CtrI-U          | 删除当前行的所有文本                |
| Ctrl-F          | 光标向后移动一个字符                |
| Ctrl-B          | 光标向前移动一个字符                |
| CtrI-L          | 清空屏幕                            |

Jupyter notebook快捷键：

| 快捷键      | 说明                      |
| ----------- | ------------------------- |
| M           | switch to markdown        |
| A           | create celIs above        |
| B           | create cells below        |
| D+D         | delete cell               |
| Y           | switch to code            |
| Ctrl+Enter  | run cell and stay on cell |
| Shift+Enter | run cell and go to next   |

### 魔术命令

IPython中特殊的命令（Python中没有）被称作“魔术”命令。这些命令可以使普通任务更便捷，更容易控制IPython系统。魔术命令是在指令前添加百分号%前缀。例如，可以用`%timeit`（这个命令后面会详谈）测量任何Python语句，例如矩阵乘法，的执行时间：

```python
In [20]: a = np.random.randn(100, 100)

In [20]: %timeit np.dot(a, a)
10000 loops, best of 3: 20.9 µs per loop
```

魔术命令可以被看做IPython中运行的命令行。许多魔术命令有“命令行”选项，可以通过？查看：

```python
In [21]: %debug?
Docstring:
::

  %debug [--breakpoint FILE:LINE] [statement [statement ...]]

Activate the interactive debugger.
......
```

魔术函数默认可以不用百分号，只要没有变量和函数名相同。这个特点被称为“自动魔术”，可以用`%automagic`打开或关闭。

一些魔术函数与Python函数很像，它的结果可以赋值给一个变量：

```python
In [22]: %pwd
Out[22]: '/home/wesm/code/pydata-book

In [23]: foo = %pwd

In [24]: foo
Out[24]: '/home/wesm/code/pydata-book'
```

IPython的文档可以在shell中打开，我建议你用`%quickref`或`%magic`学习下所有特殊命令。下表列出了一些可以提高生产率的交互计算和Python开发的IPython指令。

| 命令                | 说明                                                     |
| ------------------- | -------------------------------------------------------- |
| %quickref           | 显示 IPython 的快速参考。                                |
| %magic              | 显示所有魔术命令的详细文档。                             |
| %debug              | 在出现异常的语句进入调试模式。                           |
| %hist               | 打印命令的输入 (可以选择输出) 历史                       |
| %pdb                | 出现异常时自动进入调试。                                 |
| %paste              | 执行剪贴板中的代码。                                     |
| %cpaste             | 开启特别提示，手动粘贴待执行代码。                       |
| %reset              | 删除所有命名空间中的变量和名字。                         |
| %page OBJECT        | 美化打印对象，分页显示。                                 |
| %run script.py      | 运行代码。                                               |
| %prun statement     | 用 CProfile 运行代码，并报告分析器输出。                 |
| %time statement     | 报告单条语句的执行时间。                                 |
| %timeit statement   | 多次运行一条语句，计算平均执行时间。适合执行时间短的代码 |
| %who, %who ls %whos | 显示命名空间中的变量，三者显示的信息级别不同。           |
| %xdel variable      | 删除一个变量，并清空任何对它的引用。                     |

### 集成Matplotlib

IPython在分析计算领域能够流行的原因之一是它非常好的集成了数据可视化和其它用户界面库，比如matplotlib。不用担心以前没用过matplotlib，本书后面会详细介绍。`%matplotlib`魔术函数配置了IPython shell和Jupyter notebook中的matplotlib。这点很重要，其它创建的图不会出现（notebook）或获取session的控制，直到结束（shell）。

在IPython shell中，运行`%matplotlib`可以进行设置，可以创建多个绘图窗口，而不会干扰控制台session：

```python
In [26]: %matplotlib
Using matplotlib backend: Qt4Agg
```

在JUpyter中，命令有所不同（图2-6）：

```python
In [26]: %matplotlib inline
```



