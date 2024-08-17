---
tags:
  - 命令行/docker
  - 开发/容器化/Docker
---

# Docker
[中文教程](https://www.runoob.com/docker/docker-tutorial.html)

## 容器

### 容器和虚拟机的区别
当谈论容器化和虚拟机时，你可以将其比喻为两种不同的方式来“打包”和运行软件应用程序，就像在不同种类的盒子中运输货物一样。

容器化：
容器化类似于用一种魔法盒子来打包和运输你的应用程序。这个盒子包含了你的应用程序和所有需要的东西，比如库、配置文件等。不同之处在于，这个盒子非常轻便，几乎没有额外的重量，因此非常高效。
- 轻盒子： 容器非常轻便，因为它们与主机操作系统共享许多组件，只包含应用程序及其依赖项。
- 快速启动： 容器可以在瞬间启动，就像打开盒子一样迅速。
- 共享资源： 多个容器可以在同一台机器上运行，共享操作系统的资源，而互不干扰。

虚拟机：
虚拟机则像一台小型的模拟计算机。你把你的应用程序和所有东西都放在这台模拟机器中，然后在主机机器上运行它。但这个模拟机器比容器要重，因为它需要一个完整的操作系统，就像在一个大箱子中运输一个小盒子。

- 重机器： 虚拟机包含了一个完整的操作系统，所以相对较重。
- 较慢启动： 启动虚拟机通常需要更多时间，就像启动一台真正的计算机一样。
- 资源隔离： 虚拟机提供了更强的资源隔离，但也需要更多的资源。

综上所述，容器化更加轻便和高效，适用于快速部署和运行应用程序，而虚拟机提供了更严格的隔离，但通常需要更多资源和时间。你可以根据你的需求来选择使用容器化或虚拟机技术。

## Install Docker
### Ubuntu 20/22
`docker.io` 是 Docker 的一个旧版本包名，在较早的 Ubuntu 版本和一些特定的环境中使用。它在 Ubuntu 的官方存储库中存在，但通常不如 Docker 官方的安装方法更新。

旧版本
```shell
sudo nala install docker
sudo nala install docker.io
```

新版本
```shell
sudo nala update
sudo nala install ca-certificates curl gnupg lsb-release

# 添加 Docker 官方 GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 设置 Docker 稳定版存储库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo nala update
sudo nala install docker-ce docker-ce-cli containerd.io

# 将当前用户添加到 Docker 组
sudo usermod -aG docker $USER
```

安装完之后建议重启或注销一次

启动并验证 Docker：
```shell
sudo systemctl start docker
sudo docker run hello-world
```

第一次报错可以尝试重启一次
```shell
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 基本命令
### 查看版本
`docker --version`

### 查看状态
`sudo systemctl status docker`

### 查看容器
查看运行中的容器：`docker ps`

列出所有容器（包括已停止的）：`docker ps -a`

输出含义:

例如如下输出
```bash
CONTAINER ID   IMAGE              COMMAND                  CREATED         STATUS        PORTS  NAMES
f9a15ae35170   nvidia/cuda:v1     "/opt/nvidia/nvidia_…"   12 months ago   Up 5 months           chatglm_container
```

- IMAGE: This specifies the Docker image used to create the container. In this case, it's nvidia/cuda with the tag v1. The image provides the environment and application that the container runs.
- COMMAND: This shows the command that is being executed when the container starts. The full command can be viewed with `docker inspect` or `docker ps -a --no-trunc`. The ellipsis (`…`) indicates that the command is truncated for display purposes.
- PORTS: This field lists any ports that are mapped between the container and the host machine. If the container had exposed ports, you would see entries like `0.0.0.0:80->80/tcp`, indicating port forwarding rules. In this entry, no ports are listed, meaning the container does not have any exposed ports or they are not shown in this truncated view.
- NAMES: This is the name given to the container. Docker allows you to assign a name to each container, making it easier to manage and reference them compared to using the container ID. If no name is assigned, Docker generates a random one.

### 列出所有镜像
`docker images`

### 查看容器日志
To view the logs of the `qanything-container-local` container:
```shell
sudo docker logs qanything-container-local
```

### 删除容器
`docker rm <container_id>`

### 删除镜像
`docker rmi <image_id>`

## Docker for Ultralytics YOLO

### Setting Up

1. [Install Docker](#install-docker)
2. Install NVIDIA Docker Runtime （可选）
```shell
sudo nala update
sudo nala install -y nvidia-docker2
sudo systemctl restart docker
```
3. Pull the Ultralytics Docker Image:
`docker pull ultralytics/ultralytics:latest`

4. Run the Docker Container:
`docker run -it --gpus all --ipc=host ultralytics/ultralytics:latest`

没有GPU的情况可以：`docker run -it --ipc=host ultralytics/ultralytics:latest`

### Example Commands for Running YOLOv8 in Docker
- `-v` 后的具体目录需要指定为工程的运行目录
- `model=` 后面加工程的运行目录下模型存储的相对路径，`source`同理。

1. Object Detection:
```shell
docker run -it --rm -v ~/knoll/code_base/cv/cv_test:/yolov8 ultralytics/ultralytics:latest yolo detect predict save model=/yolov8/models/yolov8s.onnx source=/yolov8/test_data/00255-3709288448.png
```

2. Image Segmentation:
```shell
docker run -it --rm -v ~/yolov8:/yolov8 ultralytics/ultralytics:latest yolo segment predict save model=yolov8s-seg.pt source=inputs/test.jpg
```

3. Image Classification:
```shell
docker run -it --rm -v ~/yolov8:/yolov8 ultralytics/ultralytics:latest yolo classify predict save model=yolov8s-cls.pt source=inputs/test.jpg
```

## Nvidia Docker
https://hub.docker.com/r/nvidia/cuda/

### Install
#### 直接拉取镜像
在docker页面的Tags选项卡中有最新版本的各个系统的容器命令，
如在Ubuntu20的主机中，安装cuda12.4容器(注意你的GPU驱动允许的最高cuda版本，否则可能搭建环境不成功)：
```shell
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
```
镜像`nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04`是一个包含 CUDA 和 cuDNN 开发环境的 Docker 镜像。具体说明如下：

1. CUDA 版本: 12.4.1
包含了 CUDA 12.4.1 版本的工具包，这是一套用于开发并行计算应用的工具和库，特别是针对 NVIDIA GPU。
2. cuDNN:
cuDNN (CUDA Deep Neural Network library) 是 NVIDIA 提供的一个用于深度学习的 GPU 加速库。这个库在训练和推理阶段能大大提高卷积神经网络的性能。这个镜像包含 cuDNN，可以直接用于开发和运行深度学习应用。
3. 开发环境 (devel):
devel 表示这是一个开发版本的镜像，包含了完整的开发工具链，例如 CUDA 编译器 (nvcc)、cuDNN 库、其他 CUDA 库（如 cuBLAS、cuFFT 等），以及各种示例代码和调试工具。
适合在容器中进行 CUDA 应用程序的开发、编译和测试。
4. Ubuntu 版本: 20.04
这个镜像是基于 Ubuntu 20.04 LTS 构建的。Ubuntu 20.04 是一个长期支持版本，适合用于生产环境。

#### 拉取失败了？
方法1：使用老毛子固件的路由配置ShadowSocks，或经过ROOT后的安卓手机/有无线收发模块安卓嵌入式设备使用【VPN热点】APP和Clash分享WiFi。

方法2: 用Google Colab下载离线包。
1. 首先应用到这个工具 https://github.com/drengskapur/docker-in-colab 按照提示在Colab中创建一个cell
```shell
# Copyright 2024 Drengskapur
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title {display-mode:"form"}
# @markdown <br/><br/><center><img src="https://cdn.jsdelivr.net/gh/drengskapur/docker-in-colab/assets/docker.svg" height="150"><img src="https://cdn.jsdelivr.net/gh/drengskapur/docker-in-colab/assets/colab.svg" height="150"></center><br/>
# @markdown <center><h1>Docker in Colab</h1></center><center>github.com/drengskapur/docker-in-colab<br/><br/><br/><b>udocker("run hello-world")</b></center><br/>
def udocker_init():
    import os
    if not os.path.exists("/home/user"):
        !pip install udocker > /dev/null
        !udocker --allow-root install > /dev/null
        !useradd -m user > /dev/null
    print(f'Docker-in-Colab 1.1.0\n')
    print(f'Usage:     udocker("--help")')
    print(f'Examples:  https://github.com/indigo-dc/udocker?tab=readme-ov-file#examples')

    def execute(command: str):
        user_prompt = "\033[1;32muser@pc\033[0m"
        print(f"{user_prompt}$ udocker {command}")
        !su - user -c "udocker $command"

    return execute

udocker = udocker_init()
```

2. 创建第2个cell
```shell
def save_image(image_name):
  udocker("pull " + image_name)
  file_name = image_name.replace(":", "_")
  file_name = file_name.replace("/", "_")+'.tar'
  udocker("save -o "+ file_name+" " + image_name)
  !gzip -c /home/user/{file_name} > /content/{file_name}.gz
```

3. 创建第3个cell
```shell
save_image("nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04")
```

4. 下载后本地读取
```shell
docker load -i nvidia_cuda_12.4.1-cudnn-devel-ubuntu20.04.tar.gz
```

#### 使用镜像
[查看已经安装的镜像](#列出所有镜像)

使用GPU资源运行容器
```shell
docker run -it --gpus all --name my_cuda_container nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 /bin/bash
```
- `-it`：表示交互模式运行容器。
- `--name my_cuda_container`：指定容器的名字，你可以根据需要更改。
- `/bin/bash`：指定要运行的命令，这里是启动一个 Bash shell。
- `--gpus all`：表示容器可以访问所有可用的 GPU。

启动一个后台守护进程
```shell
docker run -d --gpus all --name my_cuda_container nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 tail -f /dev/null
```
这会启动容器并运行`tail -f /dev/null`，保持容器运行而不执行任何实际任务。
你可以之后通过`docker exec -it my_cuda_container /bin/bash`进入容器。

挂载本地目录（可选）
```shell
docker run -it --gpus all --name my_cuda_container -v /path/on/host:/path/in/container nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 /bin/bash
# 后台运行
docker run -d --gpus all --name chat_tts_cu124 -v /mnt/knoll/chat_tts:/home/knoll/chat_tts nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04 tail -f /dev/null
```
`/path/on/host:/path/in/container`：将主机的路径`/path/on/host`挂载到容器中的`/path/in/container`。

带端口映射的
```shell
docker run -d --name my_flask_container -p 6300:6300 nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
```
`-p 6300:6300`: Maps port 6300 on the host machine to port 6300 in the container. Unfortunately, once a container is created without the port mapping, you can't add it later without recreating the container. So you'll need to remove the existing container and create a new one with the correct port mapping.

```shell
apt-get install -y curl unzip python3 python3-pip git
apt-get install -y git vim
apt update && apt upgrade
```

#### 安装Conda
1. 手动把安装包放到挂载的目录下，然后进到容器里安装。安好后`exit`然后`docker exec -it my_cuda_container /bin/bash`重进一下

安装好后，如果创建环境卡住了可以换源，参考【终端】-【anaconda】-【换源】

#### 安装驱动
查看那些版本可用
```shell
apt search nvidia-driver
```

安装
```shell
apt install nvidia-driver-<version>
```
