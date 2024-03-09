---
tags:
  - 调试日志
---

# Log

## Python

### Gradio

#### Gradio和Clash代理冲突

- 方法1:

按照错误提示修改gr库
`routes.py` line117
```python
#client = httpx.AsyncClient()
```

此时，import gradio不报错了。另一个文件requests库的 `adapter.py` line 340
```python
proxy=None
```

- 方法2:
```python
os.environ["all_proxy"] = ""
```

### Pyrender

OpenGL报错
https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris/75333466#75333466
```shell
conda install -c conda-forge libstdcxx-ng
```

### QT
```shell
pip install pyqt5
```

## Stable Diffusion
### 环境
CUDA 11.7 (nvcc -V) + Python 3.9.17
```
clip                      1.0
gradio                    3.32.0
jsonfiler                 0.0.3
numpy                     1.23.5
open-clip-torch           2.20.0
opencv-python             4.8.0.74
pytorch-lightning         1.9.4
scikit-learn              1.3.0
timm                      0.6.7
torch                     2.0.1+cu118
torchaudio                没安
torchdiffeq               0.2.3
torchmetrics              1.0.3
torchsde                  0.2.5
torchvision               0.15.2+cu118
tqdm                      4.65.0
```
CUDA 11.6 + Python 3.9.17 + 515.65.01 + 5.15.0-46-generic
```
clip                      1.0
gradio                    3.32.0
numpy                     1.23.5
open-clip-torch           2.7.0
opencv-contrib-python     4.8.0.76
opencv-python             4.8.0.76
pytorch-lightning         1.9.4
scikit-learn              1.3.0
timm                      0.6.7
torch                     2.0.1
torchaudio                2.0.2
(pip install torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu116)
torchdiffeq               0.2.3
torchmetrics              1.0.3
torchsde                  0.2.5
torchvision               0.15.2
tqdm                      4.66.1
```
pip install torch==2.0.1 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu116

报错

TypeError: AsyncConnectionPool.__init__() got an unexpected keyword argument 'socket_options'
https://qiita.com/bigmon/items/1a6f220df98941c81f94

### TensorRT加速

按照【TensorRT安装】章节安装TensorRT

`models/Unet-onnx`存入onnx模型，
`models/Unet-trt`存入trt模型

### ComfyUI

安装主程序
```
https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 main.py #可以改文件里的checkpoint路径为模型路径
```


Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae

插件管理器
```shell
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

## Linux

### Ubuntu 环境
- 内核: 5.15.0-46-generic (uname -r)
- 显卡驱动: 515.65.01 (默认预装)
- cuda: https://developer.nvidia.com/cuda-toolkit-archive

教程https://blog.csdn.net/takedachia/article/details/130375718

```shell
$ vim ~/.bashrc
export PATH=$PATH:/usr/local/cuda-12.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.2/lib64
$ source ~/.bashrc
$ nvcc -V
```

- cudnn
```shell
sudo cp cudnn-linux-x86_64-8.9.2.26_cuda11-archive/include/cudnn.h    /usr/local/cuda-11.6/include
sudo cp cudnn-linux-x86_64-8.9.2.26_cuda11-archive/lib/libcudnn*    /usr/local/cuda-11.6/lib64
sudo chmod a+r /usr/local/cuda-11.6/include/cudnn.h   /usr/local/cuda-11.6/lib64/libcudnn*
echo "安完了"
```

- 应用安装
```shell
  sudo dpkg -i [安装包名称].deb
  sudo apt install terminator
  sudo apt install gcc
```

**Anaconda安装后的操作**
```shell
  sudo gedit .bashrc
  export PATH=[路径]/anaconda3/bin:$PATH
  source ~/.bashrc
  conda env list
  conda create -n drawer python=3.9
  conda activate drawer
  # pip换源
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  https://mirrors.aliyun.com/pypi/simple

  pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
### 同时驱动多屏
设置-显示器里显示的图标是代表每个屏幕的实际位置。可以用鼠标调节。

OpenCV创建一个窗口，然后move到那个屏幕上，然后全屏可以向指定屏幕投屏。

### 换源
```shell
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo vim /etc/apt/sources.list
```

配置内容如下
这里加`[arch=amd64]`是因为阿里云报了个错误。
```
deb [arch=amd64] http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb [arch=amd64] http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb [arch=amd64] http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb [arch=amd64] http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse

# deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
# deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
# deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
# deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse

## Pre-released source, not recommended.
# deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
```

```shell
sudo apt-get update
```

为什么是`[arch=amd64]`？
`dpkg --print-architecture`
后显示arm64
```
# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://ports.ubuntu.com/ubuntu-ports jammy main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy main restricted

## Major bug fix updates produced after the final release of the
## distribution.
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-updates main restricted

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://ports.ubuntu.com/ubuntu-ports jammy universe
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy universe
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
deb http://ports.ubuntu.com/ubuntu-ports jammy multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-updates multiverse

## N.B. software from this repository may not have been tested as
## extensively as that contained in the main release, although it includes
## newer versions of some applications which may provide useful features.
## Also, please note that software in backports WILL NOT receive any review
## or updates from the Ubuntu security team.
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-backports main restricted universe multiverse

deb http://ports.ubuntu.com/ubuntu-ports jammy-security main restricted
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-security main restricted
deb http://ports.ubuntu.com/ubuntu-ports jammy-security universe
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-security universe
deb http://ports.ubuntu.com/ubuntu-ports jammy-security multiverse
# deb-src http://ports.ubuntu.com/ubuntu-ports jammy-security multiverse
~
```

### Armbian 换源

https://mirrors.tuna.tsinghua.edu.cn/help/armbian/

**配置armbian.list**

编辑 /etc/apt/sources.list.d/armbian.list，将 http://apt.armbian.com 替换为以下链接
```
https://mirrors.tuna.tsinghua.edu.cn/armbian
```

`sudo apt update`后
```
错误:3 https://mirrors.tuna.tsinghua.edu.cn/armbian jammy InRelease
  由于没有公钥，无法验证下列签名： NO_PUBKEY 93D6889F9F0E78D5
```
之后在提示的https://mirrors.tuna.tsinghua.edu.cn/armbian/ 找到`armbian.key`文件

如果你找到了一个名为 armbian.key 的文件，这是公钥文件。你可以使用以下命令将该公钥导入到 APT 密钥环中：

```shell
sudo apt-key add armbian.key
```
或者，使用 gpg 命令手动导入：

```shell
gpg --import armbian.key
```
完成后，再次运行 sudo apt update，检查问题是否已解决。请记住，APT 密钥服务器可能会有变化，所以你可能需要定期更新密钥或者使用其他可用的密钥服务器。

**配置sources.list**
```shell
> sudo vim /etc/apt/sources.list
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse
#deb-src http://ports.ubuntu.com/ jammy main restricted universe multiverse

deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse
#deb http://ports.ubuntu.com/ jammy-security main restricted universe multiverse
#deb-src http://ports.ubuntu.com/ jammy-security main restricted universe multiverse

deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse
#deb http://ports.ubuntu.com/ jammy-updates main restricted universe multiverse
#deb-src http://ports.ubuntu.com/ jammy-updates main restricted universe multiverse

deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse
#deb http://ports.ubuntu.com/ jammy-backports main restricted universe multiverse
#deb-src http://ports.ubuntu.com/ jammy-backports main restricted universe multiverse
```

#### Using nala
```shell
sudo nala fetch
```

### Labelimg
https://github.com/HumanSignal/labelImg
```shell
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py
python3 labelImg.py [IMAGE_PATH] [PRE-DEFINED CLASS FILE]
```
或（Py3.8测试成功）
```shell
pip3 install labelImg
sudo nala install pyqt5-dev-tools
labelImg
```

### Nala
gitlab: https://gitlab.com/volian/nala

#### Ubuntu 20.04
同样适用于Rock3a的官方5.10linux kernel的Debian系统

支持架构
- x86_64
- aarch64

https://askubuntu.com/questions/1422002/unable-to-correct-problems-you-held-broken-packages-nala

https://forums.linuxmint.com/viewtopic.php?t=390049

在此之前要更新仓库，前往https://gitlab.com/volian/volian-archive/-/releases 下载`volian-archive-keyring_0.1.0_all.deb`和`volian-archive-nala_0.1.0_all.deb`
```shell
sudo apt install ./volian-archive*.deb
```

```shell
sudo apt update && sudo apt install nala-legacy
```
之后需要和sudo一起使用才能找到nala

之后提示报错暂且可以不管
```
neardi@LPA3568:~/nala/nala/nala$ sudo nala install python3-pip
Error: [Errno 2] No such file or directory: '/etc/nala/nala.conf'
Notice: Unable to read config file: /etc/nala/nala.conf. Using defaults
```

提示找不到libpython3.9
```shell
sudo add-apt-repository ppa:deadsnakes/ppa
```

#### Ubuntu 22

适用于
- Rock5b 的第三方Ubuntu22镜像
- Armbian 23.8.1 Jammy with Linux 5.10.160-legacy-rk35xx
- Ubuntu 22.04 LTS(新系统需要fix broken一下)

在此之前要更新仓库，前往https://gitlab.com/volian/volian-archive/-/releases 下载`volian-archive-keyring_0.1.0_all.deb`和`volian-archive-nala_0.1.0_all.deb`
```shell
sudo apt install ./volian-archive*.deb
sudo apt update && sudo apt install nala -y && sudo nala upgrade -y
```

## TensorRT安装
https://zhuanlan.zhihu.com/p/159591904
1. 直接去Nidia的TensorRT官网下载tar压缩包
2. 解压到`stable-diffuextension-webui/extensions/stable-diffusion-webui-tensorrt-master`下，单独作为一个文件夹。并且不要套娃。
如`xxx/stable-diffuextension-webui/extensions/stable-diffusion-webui-tensorrt-master/TensorRT-8.6.1.6`
```shell
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
```
3. 进入tensorRT目录下的Python目录，安装包
```shell
cd TensorRT-8.6.1.6/python
sudo pip install tensorrt-8.6.1-cp39-none-linux_x86_64.whl
```
4. 将其添加到环境变量.bashrc中
```shell
vim ~/.bashrc
```
添加以下内容
```shell
export LD_LIBRARY_PATH=/path/to/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/TensorRT-8.6.1.6/lib:$LIBRARY_PATH
```
刷新
```shell
source ~/.bashrc
```

中途没安装pycuda库

### YOLOv8 & TensorRT
Requirements:
- [Ultralytics库](https://github.com/ultralytics/ultralytics): `pip install ultralytics`

TensorRT & YOLOv8 教程：https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/

1. 安装好Ultralytics库，后会出现新命令`yolo`。先下载一个`.pt`模型，然后使用命令生成`.engine`模型:
```shell
yolo export model=yolov8n-seg.pt format=engine half=True device=0
```
2. 之后把代码中的`.pt`模型直接改成`.engine`模型即可。
```python
from ultralytics import YOLO
# Load a model
model_path = 'yolov8n-seg.engine'  # model_path = 'yolov8n-seg.pt'
model = YOLO(model_path)
```

## Clash
socks主机 127.0.0.1 7890
```shell
cp Country.mmdb config.yaml ~/.config/clash
```
dashboard端口:9090。进去后点下add

http://clash.razord.top/

## 虚拟机
### 文件互传
用SSH
```shell
sudo apt-get install openssh-server
```

## 单片机
### Jlink
去[SEGGER官网](https://www.segger.com/downloads/jlink/)下载最新的Jlink驱动安装

- Cannot read Jlink version number, 啥也选不了

  打开设备管理器，发现未知设备

  点更新驱动，然后浏览本地搜索，路径C:ProgramFIles/SEGGER/。搜索到驱动后正常

- JLink能读取到ID，但是右边JTAG Device Chain不显示芯片

  Port选SW，把MAX调低
