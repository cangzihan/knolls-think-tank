# MOT

## deepsort

docker文件写法
```text
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    bash \
    bash-completion \
    vim \
    less \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# 设置默认 shell 为 bash
SHELL ["/bin/bash", "-c"]

# 设置彩色 ls + 更友好的 PS1
RUN echo "alias ls='ls --color=auto'" >> ~/.bashrc && \
    echo "PS1='\u@\h:\w\$ '" >> ~/.bashrc

WORKDIR /mot_hub
COPY . .
COPY tensorflow-2.20.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl .

RUN pip install tensorflow-2.20.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install --no-cache-dir -r requirements.txt
RUN rm tensorflow-2.20.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# 设置容器默认 SHELL
SHELL ["/bin/bash", "-c"]

```


