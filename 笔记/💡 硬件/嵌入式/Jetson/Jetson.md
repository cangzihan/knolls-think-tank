---
tags:
  - 机器视觉
  - Computer Vision
  - YOLO
---
# Jetson

控制面板
```shell
jtop
```

查看型号
```shell
cat /proc/device-tree/model
```

查看L4T版本
```shell
head -n 1 /etc/nv_tegra_release
```

查看Jetpack版本
```shell
dpkg -l | grep 'nvidia-jetpack'
```

40pin引脚
```shell
sudo /opt/nvidia/jetson-io/jetson-io.py
```

安装方法可以执行`00`的命令。

`01`命令是调出引脚的设置界面，在这里也可以查看当前的引脚定义。

引脚实物的排列方式（把40pin正对着自己，总不可能有人把Nvidia盒子倒着放吧）：
适用于（Jetson AGX Xavier 32G）
| 39 | 37 | 35  | 33  | 31 | 29 | 27  | 25 | 23 | 21  | 19  | 17 | 15 | 13  | 11  | 9 | 7  | 5  | 3  |  1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---   |
| 40 | 38 | 36 | 34 | 32  | 30 | 28  | 26  | 24  | 22  | 20 | 18  | 16  | 14 | 12  | 10  | 8  | 6 | 4  | 2    |

引脚说明：

|编号|功能|编号|      功能      |
| --- | :---: | --- |:------------:|
|1|3.3V|2|      5V      |
| 3 | i2c8 |   4 |      5V      |
| 5 | i2c8 |   6 |     GND      |
| 7 | unused |   8 | uarta（可能是TX） |
| 9 |  GND |  10 | uarta（可能是RX） |
| 11 |  uarta |  12 |    unused    |
| 13 | pwm8 |  14 |     GND      |
| 15 | unused |  16 |    unused    |
| 17 | 3.3V |  18 |     pwm5     |
| 19 | unused |  20 |     GND      |
| 21 | unused |  22 |      NA      |
| 23 | unused |  24 |    unused    |
| 25 |  GND |  26 |    unused    |
| 27 | i2c2 |  28 |     i2c2     |
| 29 | unused |  30 |     GND      |
| 31 | unused |  32 |    unused    |
| 33 | unused |  34 |     GND      |
| 35 | unused |  36 |    uarta     |
| 37 | unused |  38 |    unused    |
| 39 |  GND |  40 |    unused    |

查看好版本后，去官网下载PyTorch wheel: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
手动安装PyTorch

顶楼还有torchvision的安装方法
```shell
# 实际使用时注意版本
git clone --branch release/0.13 https://github.com/pytorch/vision torchvision
```

在Jetson上
```shell
# 安装不了libopenblas-dev也没问题
cd torchvision
export BUILD_VERSION=0.13.0
conda activate vision8
python3 setup.py install --user #提示报错，https://github.com/pytorch/vision/pull/6141
vim torchvision/csrc/ops/quantized/cpu/qnms_kernel.cpp
#去掉那行代码 //#include <ATen/native/quantized/affine_quantizer.h>
python3 setup.py install --user

# AVXXXX/FFmpeg报错。卸载天杀的ffmpeg,安完torchvision再装回来
sudo rm -rf /usr/local/bin/ffmpeg /usr/local/include/*ffmpeg* /usr/local/lib/*ffmpeg* /usr/local/share/*ffmpeg* /usr/local/share/man/man1/ffmpeg.1
```

版本对应关系

| `torch`            | `torchvision`      | Python              |
| ------------------ | ------------------ | ------------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11`   |
| `2.1`              | `0.16`             | `>=3.8`, `<=3.11`   |
| `2.0`              | `0.15`             | `>=3.8`, `<=3.11`   |
| `1.13`             | `0.14`             | `>=3.7.2`, `<=3.10` |


## SSH投屏
Tabby中，找到连接配置，然后高级设置，X11 转发打开

## FFmpeg
直接用nala安装提示找不到，那么就通过源码编译

在 https://github.com/FFmpeg/FFmpeg.git 下载源码

```shell
unzip FFmpeg-master.zip
cd FFmpeg-master
./configure
make
sudo make install

#验证
ffmpeg -version
```

## 声音
```shell
cat /proc/asound/
```

## 串口
https://blog.csdn.net/weixin_44350337/article/details/111623475
### 调试
Ubuntu 22自带CH341驱动，
```shell
# 替换自己的内核目录
ls /lib/modules/6.2.0-37-generic/kernel/drivers/usb/serial/
# ch341.ko存在
```
但每次插上会自动disconnect,解决方法：
https://stackoverflow.com/questions/70123431/why-would-ch341-uart-is-disconnected-from-ttyusb
```shell
sudo apt remove brltty
```
BRLTTY 是一个用于盲人和视障人士的屏幕阅读软件，它使他们能够通过串口等方式与计算机进行交互。


pip install websocket-client
