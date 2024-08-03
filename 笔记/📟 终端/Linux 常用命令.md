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
```t
