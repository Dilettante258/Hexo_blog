---
title: Ubuntu上的Ruby和Jekyll安装记录
categories: Linux
date: 2023-12-7 12:27:23
mathjax: true
---

# Prerequisites

先检查是否有安装依赖项，按照 [requirements](https://jekyllrb.com/docs/installation/#requirements) 提示。

我是Ubuntu系统，根据[这个](https://jekyllrb.com/docs/installation/ubuntu/)的提示安装依赖项。

```
sudo apt update
sudo apt upgrade
sudo apt install gcc g++ make curl autoconf bison build-essential libssl-dev libyaml-dev libreadline6-dev zlib1g-dev libncurses5-dev libffi-dev libgdbm6 libgdbm-dev libdb-dev
```

# 安装 Ruby

这是官方教程：[链接](https://jekyllrb.com/docs/installation/#requirements)

简单地按照 Jekyll 的教程安装只能安装2.7的Ruby，而需要的一个`sass-embedded`需要更高版本的Ruby。

Ruby的[官网](https://www.ruby-lang.org/en/downloads/)上这么说的：在Linux/UNIX上，您可以使用您的发行版或第三方工具的包管理系统（[rbenv](https://github.com/rbenv/rbenv)和[RVM](http://rvm.io/))。

所以，运行以下命令将 Ruby shell 脚本文件安装程序下载并运行：

```
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash
```

接下来，使用以下 shell 脚本命令更新路径环境：

```
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

然后可能需要重启一下终端。

输入以下命令安装Ruby，并在全局激活：

```
rbenv install 3.2.2
rbenv global 3.2.2
```

![image-20231207155627856](http://106.15.139.91:40027/uploads/2312/658d4c60513fd.png)

```
sudo apt-get install ruby-full build-essential zlib1g-dev
```

需要避免以 root 用户身份安装 RubyGems 软件包（称为 gems）。相反 为您的用户账户设置 GEM 安装目录。输入以下命令将GEM 安装路径的环境变量添加到要配置的文件`~/.bashrc`中 ：

```
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

换个源，然后安装`sass`，不然会报错：

```
gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
gem update --system
gem install sass-embedded -v 1.63.6
```

最后，安装 Jekyll 和 Bundler：

```
gem install jekyll bundler
```

现在就准备好开始使用 Jekyll了。

# 创建基础站点

1. 下列命令在`./myblog`中创建了 Jekyll 站点。

   ```
   jekyll new myblog
   ```

2. 切换到新目录。

   ```
   cd myblog
   ```

3. 生成站点并使其在本地服务器上可用。

   ```
   bundle exec jekyll serve
   ```

4. 浏览到 [http://localhost:4000](http://localhost:4000/)



后面没玩这个了，因为觉得有点老。用Astro搭建了个人学术主页。

