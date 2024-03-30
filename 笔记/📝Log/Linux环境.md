
# Linux 环境

如果是远程装环境，则需要先装SSH服务
```shell
sudo apt-get install openssh-server
```

```shell
sudo apt install ./volian-archive*.deb
echo "deb-src https://deb.volian.org/volian/ scar main" | sudo tee -a /etc/apt/sources.list.d/volian-archive-scar-unstable.list
sudo apt update && sudo apt install nala -y && sudo nala upgrade -y

chmod a+x Anaconda3-2022.05-Linux-x86_64.sh
./Anaconda3-2022.05-Linux-x86_64.sh

sudo nala install vim
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

`requirements.txt`
```
pydub
pyserial
zmq
```
