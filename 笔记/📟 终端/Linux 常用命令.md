---
tags:
  - Linux
  - Ubuntu
  - Debian
---

# Linux 常用命令

## 系统信息
内核版本
```shell
uname -r
```

## 查看内存
```shell
free -m
```

## 查看存储空间
```shell
df -h
```

## 安Python
sudo 里的Python安装pip
```shell
sudo nala install python3-pip
```

## 查看占用端口的进程，并关闭
```shell
sudo lsof -i :<端口号>
sudo kill -9 <PID>
```

## 查看相机
```shell
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

## 自动输入密码
```shell
echo 123456 | sudo -S /path/to/my_script.sh
```

## GPU分配
```shell
# 不分配
CUDA_VISIBLE_DEVICES="" XXX
# 分配0卡
CUDA_VISIBLE_DEVICES=0 XXX
# 分配多卡
CUDA_VISIBLE_DEVICES=0,1 XXX
```

## SSH
SSH是Secure Shell（安全外壳）的简称，是一种在不安全的网络环境中，通过加密机制和认证机制，实现安全的远程访问以及其他网络服务的安全协议。

## gcc
### 基本命令
查看版本
```shell
gcc --version
```

### 编译
利用gcc工具可以编译C语言程序

```shell
# 编译
gcc main.c -o main
# 运行
./main
```

## 开机自启动
### 使用 systemd 配置自启动服务
`systemd` 是目前主流的服务管理工具，可以通过配置文件模拟一个完整的终端环境。

#### 1. 创建一个 systemd 服务文件
使用以下命令创建服务文件：
```shell
sudo vim /etc/systemd/system/my_service.service
```
内容示例：
```text
[Unit]
Description=My Service
After=network.target

[Service]
User=你的用户名
WorkingDirectory=/path/to/your/project
ExecStart=/bin/bash -c "source /home/你的用户名/.bashrc && source /opt/ros/humble/setup.bash && python3 your_script.py"
Environment="RID=01"
Restart=always

[Install]
WantedBy=multi-user.target
```
- User: 替换为你运行程序的用户名。
- WorkingDirectory: 替换为脚本所在目录。
- ExecStart: 使用 /bin/bash -c 运行一个终端命令，加载完整的 shell 环境。
- Environment: 设置必要的环境变量。

#### 2. 启用并启动服务
```shell
sudo systemctl daemon-reload
sudo systemctl enable my_service.service
sudo systemctl start my_service.service
```

#### 3. 检查服务状态
```shell
sudo systemctl status my_service.service
```
