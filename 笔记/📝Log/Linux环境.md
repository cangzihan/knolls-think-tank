
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
```

## 显卡（N卡）
### 驱动
查看显卡驱动/使用情况
```shell
nvidia-smi
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

```shell
vim ~/.bashrc
```

虚拟环境装Cuda: https://anaconda.org/nvidia/cuda
```shell
conda install nvidia/label/cuda-12.4.0::cuda
conda install anaconda::cudnn # https://anaconda.org/conda-forge/cudnn
```

### CUDNN
安装直接查看官网教程即可
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn

sudo apt-get -y install cudnn-cuda-12
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
