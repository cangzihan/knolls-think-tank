---
tags:
  - 机器视觉
  - Computer Vision
---

# Motion

https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md

## slowfast算法
[Paper](https://arxiv.org/abs/1812.03982v1) | [Code](https://github.com/facebookresearch/SlowFast)

```shell
git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
cd slowfast
python3 setup.py build develop
pip install iopath simplejson av pytorchvideo

pip install 'git+https://github.com/facebookresearch/fairscale'

git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .

git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

numpy 1.23.5

https://github.com/Whiffe/yolov5-slowfast-deepsort-PytorchVideo
