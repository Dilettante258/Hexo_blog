---
title: 双系统Linux下Windows的NTFS盘只读解决办法
categories: Linux
date: 2023-11-9 15:30:23
---

首先尝试修复文件系统


```bash
$ sudo ntfsfix /dev/nvme0n1p3
[sudo] dilettante 的密码： 
Mounting volume... Windows is hibernated, refused to mount.
FAILED
Attempting to correct errors... 
Processing $MFT and $MFTMirr...
Reading $MFT... OK
Reading $MFTMirr... OK
Comparing $MFTMirr to $MFT... OK
Processing of $MFT and $MFTMirr completed successfully.
Setting required flags on partition... OK
Going to empty the journal ($LogFile)... OK
Windows is hibernated, refused to mount.
Remount failed: Operation not permitted
```

或者用图形化界面，打开“磁盘”（找不到直接可以搜索）：

![image-20231110163804566](http://106.15.139.91:40027/uploads/2312/658d4d766b8bf.png)

注意到，上面提示`Windows is hibernated, refused to mount. Remount failed: Operation not permitted`。这是因为Windows可能没有完全关闭，应该进入Windows系统下，关闭快速启动。具体方法请自行百度。

但是在没有开启“快速启动”的方法下，可能仍然会遇到这种情况，比如我。

![image-20231110132905370](http://106.15.139.91:40027/uploads/2312/658d4d76c6ad9.png)

看到红框标出的位置，一个是设备，一个是挂载点，输入代码

```bash
$ sudo ntfs-3g -o remove_hiberfile /dev/nvme0n1p3 /media/dilettante/Windows
```

这时有可能会提示已经被挂载或者有应用程序正在使用它。

```bash
$ sudo ntfs-3g -o remove_hiberfile /dev/nvme0n1p3 /media/dilettante/Windows
Mount is denied because the NTFS volume is already exclusively opened.
The volume may be already mounted, or another software may use it which
could be identified for example by the help of the 'fuser' command.
```

根据提示，查出

```bash
$ fuser -m /dev/nvme0n1p3
/dev/nvme0n1p3:      59075
$ kill 59075
```

再次执行之前的代码，运行成功，只读保护就已经取消了，现在可以对盘内的文件进行修改了。

