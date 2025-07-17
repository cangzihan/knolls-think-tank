
# Linux 环境

如果是远程装环境，则需要先装SSH服务

如果接下来的操作网络不好，在国内，就可以换源
```shell
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo vim /etc/apt/sources.list
```
修改为(位置随意)
```text
deb http://XXXXXX jammy main restricted
自己去网上查一个
```


```shell
sudo apt update
sudo apt-get install openssh-server
```

```shell
sudo apt install ./volian-archive*.deb
echo "deb-src https://deb.volian.org/volian/ scar main" | sudo tee -a /etc/apt/sources.list.d/volian-archive-scar-unstable.list
sudo apt update && sudo apt install nala -y && sudo nala upgrade -y

chmod a+x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh

sudo nala install vim python3-pip git
```

如果一开始没有换源，可以用如下方式换中国内源：
```shell
sudo nala fetch
```

```shell
sudo nala install v4l-utils
```

#### Python环境创建
```shell
conda create -n py310_cv python=3.10
```

#### Python音频
```shell
sudo nala install portaudio19-dev python3-all-dev
pip install pyaudio
sudo nala install pulseaudio

sudo nala install ffmpeg
```

Ubuntu 24系统可能需要改变虚拟环境C++库

【报错】ImportError: [你的Anaconda路径如/home/xxx/anaconda3]/envs/py310_cv/lib/libstdc++.so.6: version GLIBCXX_3.4.32' not found (required by /lib/x86_64-linux-gnu/libjack.so.0)
```shell
conda install -c conda-forge gcc
# 或者
# If necessary, create a symbolic link to the system's libstdc++.so.6:
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 [你的Anaconda路径如/home/xxx/anaconda3]/envs/py310_cv/lib/libstdc++.so.6
```

#### 麦克风
注意：插入麦克风后，可能在设置中【声音】中的输入没有明显的选项增加（只有一个选项），但是其中一个选项会由【Internal Microphone - Built-in Audio】变为【Microphone - Built-in Audio】
同时，音频大小（Ubuntu24可看到）会在插入后变大，这时实际上是麦克已经成功链接。

#### 蓝牙
Airpods连接
1. 解禁蓝牙
```shell
sudo rmmod btusb
sleep 1
sudo modprobe btusb
sudo /etc/init.d/bluetooth restart
```

2. 安装blueman
```shell
sudo nala install blueman
sudo nala install pulseaudio-module-bluetooth
sudo service bluetooth restart
```

3. 设置模式
```shell
sudo gedit /etc/bluetooth/main.conf
```

添加一行
```text
#ControllerMode = dual # [!code --]
ControllerMode = bredr # [!code ++]
```


```shell
sudo /etc/init.d/bluetooth restart
```

#### 串口
```shell
# 查看CH340驱动，替换内核名
ls /lib/modules/6.5.0-26-generic/kernel/drivers/usb/serial/
# 卸载盲人辅助软件
sudo apt remove brltty
# 查看串口号
ls /dev/tty*
```

#### YOLO
```shell
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Mediapipe
```shell
pip install mediapipe
```

`vision.gesturerecognizer.create_from_options` 卡死:
关闭SSH的X11转发


`requirements.txt`
```
pydub
pyserial
zmq
onnx
onnxruntime
```


#### PyTorch

```shell
#pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch==2.6.0+cu126 torchvision==1.21.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
```

## 显卡（N卡）
### 驱动
查看显卡驱动/使用情况
```shell
nvidia-smi
```

检查 NVIDIA GPU 硬件是否被识别

运行以下命令查看系统是否检测到 NVIDIA GPU：
```shell
lspci | grep -i nvidia
```

检查驱动模块是否加载

验证 NVIDIA 驱动的核心模块是否加载：
```shell
lsmod | grep nvidia
```

#### 常见问题
【报错】Failed to initialize NVML: Driver/library version mismatch
  NVML library version: 535.86

  この`nvidia-smi`のerrorの原因の一つとして、NVML（NVIDIA Management Library）と NVRM（NVIDIA Resource Manager）のバージョンの不一致が挙げられます。これらのバージョンは、それぞれ以下のコマンドで確認できます。
  ```shell
  # NVMLのバージョンの確認コマンド
  cat /sys/module/nvidia/version
  # NVRMのバージョンの確認コマンド
  cat /proc/driver/nvidia/version
  ```

  解决方案：
  1.更新驱动：
  - 你需要更新 NVIDIA 驱动到匹配 NVML 库的版本。可以通过以下命令更新驱动：
  ```shell
  sudo apt update
  sudo apt install --reinstall nvidia-driver-535
  # 这么简单就更新完了
  ```
  2.重启系统：
  - 更新驱动后，重启系统以确保新的驱动版本被正确加载：
  ```shell
  sudo reboot
  ```

#### 卸载驱动并安装另一版本的驱动
1. 卸载
```shell
sudo apt-get purge nvidia-*
sudo apt-get autoremove
sudo apt-get autoclean
```
2. 安装另一版本驱动
```shell
sudo nala install nvidia-driver-550-server-open
sudo nala install nvidia-driver-560
```
3. 重启
```shell
sudo reboot
```

### Cuda
检查Cuda版本
```shell
nvcc -V
```

#### 命令行安装
https://zenn.dev/pon_pokapoka/articles/nvidia_cuda_install

在12.6新版本中可以通过如下方式更方便安装cuda工具箱
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo nala install cuda-toolkit-12-6
```


#### 虚拟环境装Cuda
https://anaconda.org/nvidia/cuda
```shell
conda install nvidia/label/cuda-12.4.0::cuda
conda install anaconda::cudnn # https://anaconda.org/conda-forge/cudnn
pip install torch==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### CUDNN
安装直接查看官网教程即可
```shell
# 只需要指定一次
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

#sudo apt-get -y install cudnn
#sudo apt-get -y install cudnn-cuda-12

sudo aptitude install cudnn9-cuda-12-6 # 可能要选两次n，直到显示要安装的版本
```

```shell
vim ~/.bashrc
```

```text
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda
```

## 系统安装

欢迎页面，选择一个语言，然后选择右边的【Install Ubuntu】(如果选择了别的语言请选择相同位置的按钮)。

键盘布局

无线，可以选择不联网

建议正常安装，并选择图形驱动

暂时没碰到需要分区的情况


## 日语输入法
通常而言默认为IBUS输入系统，可以在【设置】-【区域与语言】-【管理已安装的语言】中查看

安装必要的软件包：
```shell
sudo apt update
sudo apt install ibus ibus-mozc -y
```

设置输入法框架为默认(貌似也可以在【设置】-【区域与语言】-【管理已安装的语言】中设置)：
```shell
im-config -s ibus
```

配置IBus：
1. 打开IBus设置：
```shell
ibus-setup
```
2. 在IBus设置中，执行以下操作：
   - 点击“输入法”标签。
   - 点击“添加”按钮，搜索并选择“日本語 - Mozc”和“简体中文 - Pinyin”。
   - 确保Mozc和Pinyin输入法已被添加到输入法列表中。

重新启动输入法框架：
```shell
pkill ibus-daemon
ibus-daemon -drx &
```

(可能要重启系统)在【设置】-【键盘】中添加输入法

按<kbd>Win</kbd> + <kbd>Space</kbd>可切换输入法

## 远程桌面
### Todesk远程连接Ubuntu卡100%，以及小窗口打不开
https://blog.csdn.net/Q95470/article/details/140008314

解决方案：
```shell
sudo vim /etc/gdm3/custom.conf
```

```text
#WaylandEnable=false # [!code --]
WaylandEnable=false  # [!code ++]
```
重启系统或命令行输入reboot重启就可以啦

## Python常见库安装
### Jupyter Notebook
```shell
pip install jupyter
```

## OpenCV GPU版
OpenCV: https://github.com/opencv/opencv

OpenCV contrib: https://github.com/opencv/opencv_contrib/tree/master
```shell
# 解压
# 复制contrib文件夹至OpenCV文件
# 开vpn!!!!
sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt-get update

# 提示没有密钥sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys

sudo apt install libjasper1 libjasper-dev

sudo apt-get install build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg-dev libtiff5-dev libswscale-dev
#（libjasper-dev）

# 查看显卡算力
# https://developer.nvidia.com/cuda-gpus#compute
# 2080Ti 7.5
# p1000 6.1
# 3060 8.6
# A2000 8.6

mv build/downloads downloads
rm -rf build
mkdir build
mv downloads build/
cd build
sudo cmake -D CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ -D CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CUDA_ARCH_BIN="8.6" -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D OPENCV_GENERATE_PKGCONFIG=1 -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.5.5/modules/ ..

sudo make -j10

sudo make install
```

## Linux添加一个自定义环境变量
在 Linux 系统中，你可以通过编辑用户的 shell 配置文件来添加自定义环境变量。以下是具体步骤：

1. 确定你的 Shell：

首先，你需要知道你当前使用的 Shell 是哪一个。常见的 Shell 有 `bash`、`zsh`、`fish` 等。

2. 编辑配置文件：

根据你使用的 Shell，编辑相应的配置文件。以下是一些常见 Shell 的配置文件：
  - Bash：`~/.bashrc` 或 `~/.bash_profile`
  - Zsh：`~/.zshrc`
  - Fish：`~/.config/fish/config.fish`

3. 添加环境变量：

打开配置文件后，在文件的末尾添加你的自定义环境变量。例如，如果你想添加一个名为 `MY_VARIABLE` 的环境变量，并将其值设置为 `my_value`，你可以这样写：

```bash
export MY_VARIABLE=my_value
```

4. 保存并关闭文件：
保存对配置文件的修改并关闭编辑器。

5. 使更改生效：
如果你编辑的是`~/.bashrc`或`~/.zshrc`，你需要重新加载该文件以使更改立即生效。你可以通过运行以下命令来实现：

```bash
source ~/.bashrc
```
或

```bash
source ~/.zshrc
```
如果你编辑的是 `~/.config/fish/config.fish`，你可以通过重新启动 Fish Shell 来使更改生效。

验证环境变量：
最后，你可以通过运行以下命令来验证环境变量是否已经成功添加：

```bash
echo $MY_VARIABLE
```
如果输出 `my_value`，则说明环境变量已经正确添加。

## 问题汇总
- e: 无法修正错误,因为您要求某些软件包保持现状,就是它们破坏了软件包间的依赖关系。

用aptitude

### Python类
- OpenCV的`cv2.imshow()`报错：Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

最终是这样解决：
```shell
pip uninstall opencv-python
pip install opencv-python
```

- qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "XXX/anaconda3/envs/vision/lib/python3XXX/site-packages/cv2/qt/plugins" even though it was found.

解决方法：把上面目录的`libqxcb.so`移走，需要的时候再放回来。

- 使用`ultralytics`等库之后，再调用`cv2.imshow()`出现程序卡死

解决方法: `pip uninstall av`

- 内核更新之后，`nvidia-smi`识别不到驱动的问题：
```shell
sudo apt update
sudo apt upgrade
```

### 挂载设备管理器不能直接打开的硬盘
```shell
sudo nala install ntfs-3g
# 以目标磁盘为/dev/nvme0n1p4为例
sudo mount -t ntfs-3g /dev/nvme0n1p4 /mnt
```
