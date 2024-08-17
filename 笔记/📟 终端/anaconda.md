# anaconda

```shell
conda env list
conda activate XXX
conda env remove --name your_env_name
conda create -n XXX python=3.8
```

使用 conda create 创建环境并指定路径
假设你想在 /path/to/your/env 创建一个新的Conda环境，可以使用以下命令：
```shell
conda create --prefix /path/to/your/env python=3.9
```

换源
```shell
vim ~/.condarc
```
```
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
ssl_verify: false
```
