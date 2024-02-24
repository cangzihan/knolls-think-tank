---
tags:
  - 机器视觉
  - Computer Vision
  - YOLO
---
# YOLO

https://docs.ultralytics.com/models/yolov8/

## Pose
### 关节定义
```python
KEYPOINTS_NAMES = [
    "nose",  # 0
    "eye(L)",  # 1
    "eye(R)",  # 2
    "ear(L)",  # 3
    "ear(R)",  # 4
    "shoulder(L)",  # 5
    "shoulder(R)",  # 6
    "elbow(L)",  # 7
    "elbow(R)",  # 8
    "wrist(L)",  # 9
    "wrist(R)",  # 10
    "hip(L)",  # 11
    "hip(R)",  # 12
    "knee(L)",  # 13
    "knee(R)",  # 14
    "ankle(L)",  # 15
    "ankle(R)",  # 16
]
```

![img](assets/yolo8_pose.png)

输出
```
0 [510 160]
1 [528 142]
2 [492 142]
3 [554 160]
4 [464 162]
5 [592 276]
6 [426 276]
7 [636 420]
8 [380 424]
9 [656 556]
10 [368 556]
11 [562 536]
12 [454 536]
13 [562 726]
14 [444 726]
15 [570 896]
16 [454 906]
```

图片中骨骼不全的情况
![img](assets/yolo8_half_pose.png)

```
0 [510 276]
1 [548 244]
2 [472 244]
3 [596 296]
4 [426 294]
5 [692 526]
6 [346 526]
7 [820 802]
8 [222 808]
9 [826 592]
10 [176 594]
11 [0 0]
12 [0 0]
13 [0 0]
14 [0 0]
15 [0 0]
16 [0 0]
```

可知，当没有对应骨骼时，输出为0

### 动作识别规划
使用normalize数据运算

输入向量：
$inputs = [k_5, k_6, k_7, k_8, k_9, k_{10}]$

batch设为历史中的10个数据，如果能可变任意长度那么训练数据为历史中的8-20个数据

如果某个值为0,那么就不检测。

输出向量：
动作类别`['打招呼', '过来', '停止', '无']`


### Code
```python
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8s-pose.pt')  # load an official model

# Predict with the model
results = model('body2.png')  # predict on an image

# Process results list
for result in results:
    keypoints = result.keypoints.xy.cpu().numpy()  # Keypoints object for pose outputs
    keypoints = keypoints.astype('int') * 2

    im_array = result.plot()  # plot a BGR numpy array of predictions

    # 获取图像的高度和宽度
    height, width = im_array.shape[:2]

    # 将图像放大 1 倍
    im_array = cv2.resize(im_array, (2 * width, 2 * height))

    # 遍历所有骨架
    for person_keypoints in keypoints:
        # 遍历每个骨架下的所有关键点
        for keypoint_id, point in enumerate(person_keypoints):
            print(keypoint_id, point)
            cv2.putText(im_array, str(keypoint_id), point,
                                     cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255,  0), 2)

    cv2.imshow('im_array', im_array)
    cv2.imwrite('result_pose.png', im_array)
    cv2.waitKey()
    cv2.destroyAllWindows()
```
