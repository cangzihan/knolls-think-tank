---
tags:
  - 机器视觉
  - Computer Vision
  - 图像语义分割
  - 实例分割
---
# Segment

## Segment Anything Model

[Base Model DL](https://github.com/ultralytics/assets/releases/download/v8.1.0/sam_b.pt)

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


