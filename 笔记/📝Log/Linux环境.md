
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

## 系统安装

欢迎页面，选择一个语言，然后选择右边的【Install Ubuntu】(如果选择了别的语言请选择相同位置的按钮)。

键盘布局

无线，可以选择不联网

建议正常安装，并选择图形驱动

暂时没碰到需要分区的情况






