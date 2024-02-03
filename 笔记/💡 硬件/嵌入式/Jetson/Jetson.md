---
tags:
  - 机器视觉
  - Computer Vision
  - YOLO
---
# Jetson
查看型号
```shell
cat /proc/device-tree/model
```

查看L4T版本
```shell
head -n 1 /etc/nv_tegra_release
```

查看Jetpack版本
```shell
dpkg -l | grep 'nvidia-jetpack'
```

查看好版本后，去官网下载PyTorch wheel: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
手动安装PyTorch

顶楼还有torchvision的安装方法
```shell
# 实际使用时注意版本
git clone --branch release/0.13 https://github.com/pytorch/vision torchvision
```

在Jetson上
```shell
cd torchvision
export BUILD_VERSION=0.13.0
conda activate vision8
python3 setup.py install --user #提示报错，https://github.com/pytorch/vision/pull/6141
vim torchvision/csrc/ops/quantized/cpu/qnms_kernel.cpp
#去掉那行代码 //#include <ATen/native/quantized/affine_quantizer.h>
python3 setup.py install --user

```
