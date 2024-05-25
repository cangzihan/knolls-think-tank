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

在docker页面的Tags选项卡中有最新版本的各个系统的容器命令，
如在Ubuntu20的主机中，安装cuda12.4容器：
```shell
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
```
