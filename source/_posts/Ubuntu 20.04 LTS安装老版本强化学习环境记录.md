---
title: Ubuntu 20.04 LTS安装老版本强化学习环境 gym0.19.0 记录
categories: 强化学习
date: 2023-10-18 18:15:00
---

入门机器学习的第一只拦路虎就是配置环境，一些经典教材和教程的上的那些代码都是在几年前写作的，然后呢，这些obsolete[过时的]代码也就相应需要配置那些环境，除了主要的库以外，其他的一堆库都要都要互相依赖。然后开发者写requirements.txt的时候，一般都只写版本要大于等于多少，而不写会限制于小于多少，然而实际上有些库就会在某个版本就会发生什么改变，然后一些用法就用不了了。这种情况下，有很大的概率就会在一些地方报错，就得排查哪些库他改了之后就不行了，需要做出相应的适配。

首先安装Ananconda，这里不再赘述。

### 创建环境

```bash
conda create gym019 python==3.7
conda activate gym019
```

至于为什么会是3.7，后面会解释。**其他开始之前，建议先做好换源**

### Opencv-python

首先安装`opencv`，不然其他包对这个包有依赖，安装了依然是需要重新安装老版本的。

```bash
pip3 install opencv-python==4.3.0.36
```

不想安装老版本的opencv-python那就需要安装 `opencv-python-headless`

```bash
pip install opencv-python-headless
```

不然在调用matplotlib画图时会报错，如下所示。

```python
QObject::moveToThread: Current thread (0x80d3f00) is not the object's thread (0x8063280).
Cannot move to target thread (0x80d3f00)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/dilettante/anaconda3/envs/Pytorch/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, minimal, minimalegl, offscreen, vnc, webgl.

已放弃 (核心已转储)
```

参考：[Uuntu20.04出现“qt.qpa.plugin: Could not load the Qt platform plugin “xcb“ in...已放弃 (核心已转储)”问题解决记录](https://blog.csdn.net/qq_49641239/article/details/116975588)

如果仍有问题，建议参考[Ubuntu20.04下解决Qt出现qt.qpa.plugin:Could not load the Qt platform plugin “xcb“ 问题](https://blog.csdn.net/gLare_nijianwei/article/details/128972547)

`opencv-python`依赖`numpy`，而对应的`numpy`依赖的版本要求是3.5-3.7之间。

![img](https://img-blog.csdnimg.cn/f66565116d6948f4a6b6f8584c2b33f4.png)

随后安装一系列强化学习所需要的一系列库。

### PyTorch

请参考[PyTorch-Get Start](https://pytorch.org/get-started/locally/)选择合适的安装命令。

我安装的是CPU版本的。

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Gym 和 Mujoco

需要安装0.19.0版本的以适配一些如`env.seed()`等函数。下载链接：[openai / gym](https://github.com/openai/gym/releases/tag/0.19.0)

下载并解压在一个合适的位置，请注意这些文件后面不能删掉。

先安装基础版本的试一下。

```bash
cd gym	# 到你解压的文件目录下 / 直接资源管理器中右键“在终端中打开”也可以
pip install -e .
```

期间也许需要安装一些GCC编译工具，有些也可能提前安装好了。

```bash
sudo apt-get install build-essential libgl1-mesa-dev libglew-dev libsdl2-dev libsdl2-image-dev libglm-dev libfreetype6-dev libglfw3-dev libglfw3 patchelf 
```

```bash
pip install setuptools==63.2.0
pip install cython==0.29
pip install swig
```

期间也许还会提示缺少什么什么，但是解决方案一般都可以轻松搜到。如果还有问题，欢迎联系我。

如果可以，接下来安装mujoco和mujoco-py。gym的0.19.0版本依然对应的是mujoco150的版本。

```bash
cd ~
mkdir .mujoco
wget https://www.roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip # 也许需要安装unzip
wget https://www.roboti.us/file/mjkey.txt
```

然后配置bashrc

~~~bash
# Mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/username/.mujoco/mjpro150/bin	# 替换成自己的username
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia						# 后面会用到的妙妙目录
~~~

执行一下bin目录下的simulate文件测试一下是否安装成功。

```bash
cd .mujoco/mjpro150/bin/
./simulate
```

然后回到gym解压的目录，安装完整的环境。

```bash
cd gym
pip install -e .[all]
```

此部分参考：[如何在linux中安装gym[all]](https://blog.csdn.net/qq_37921030/article/details/121305417)

如果安装不成功，则先单独安装mujoco-py，这里安装1.50版本的，先选定一个合适的文件夹：

```bash
wget https://github.com/openai/mujoco-py/archive/refs/tags/1.50.1.0.zip
unzip 1.50.1.0.zip
cd mujoco-py-1.50.1.0
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip3 install -e .
```

### 其他依赖

安装`matplotlib`等库……

```bash
conda install matplotlib
```

还有其他的需要的，自行安装。

![image-20231018190455415](https://img-blog.csdnimg.cn/2e769f32154341beb5a5367df5be1b78.png)

测试安装效果

![img](https://img-blog.csdnimg.cn/0e867f0a7eba42a1b924c4ad481004f0.png)