---
tags:
  - 机器视觉
  - Computer Vision
  - 图像语义分割
  - 实例分割
---
# Segment

<style>
html.dark .light-mode {
  display: none;
}

html.dark .dark-mode {
  display: block;
}

html:not(.dark) .light-mode {
  display: block;
}

html:not(.dark) .dark-mode {
  display: none;
}
</style>

## Segment Anything Model

[Base Model DL](https://github.com/ultralytics/assets/releases/download/v8.1.0/sam_b.pt) | [Paper](https://arxiv.org/pdf/2304.02643)

SAM2: [Demo](https://huggingface.co/spaces/junma/MedSAM2)

## 自动标注
```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
```

## DepthAnything
https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

```python
import os
os.environ["all_proxy"] = ""
from transformers import pipeline
from PIL import Image

# load pipe
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("crop.jpg")

# inference
depth = pipe(image)["depth"]

import numpy as np
depth_threshold = 100
depth_array = np.array(depth)

# 生成深度掩膜
mask = (depth_array > depth_threshold).astype(np.uint8) * 255

# 保存图像到文件
depth.save('output.jpg')
print("Done")
```

## UDUN

[Paper](https://arxiv.org/abs/2307.14052) | [Code](https://github.com/PJLallen/UDUN)

<div class="theme-image">
  <img src="./assets/UDUN.png" alt="Light Mode Image" class="light-mode">
  <img src="./assets/dark_UDUN.png" alt="Dark Mode Image" class="dark-mode">
</div>
