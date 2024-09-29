---
tags:
  - 机器视觉
  - Computer Vision
  - OpenCV
---

# Basic

## HSV图像
HSV（Hue, Saturation, Value）是一种颜色模型，它与人类对颜色的直观感受更加接近。HSV模型通常用于图像处理和计算机视觉领域，尤其是在需要根据颜色进行图像分割或对象识别的任务中。下面将详细介绍HSV模型及其与RGB模型的关系，以及引入HSV的原因。

### HSV模型
1. Hue（色调）：表示颜色的类型，如红色、绿色、蓝色等。在HSV模型中，色调用一个角度值来表示，范围从0到360度，形成一个色轮。例如，0度表示红色，120度表示绿色，240度表示蓝色。
2. Saturation（饱和度）：表示颜色的纯度或强度，即颜色中包含的灰度成分的比例。饱和度的值范围从0到1，其中0表示完全灰色，1表示完全饱和的颜色。
3. Value（亮度）：表示颜色的明暗程度，即颜色的亮度。值的范围也是从0到1，其中0表示黑色，1表示最亮的颜色。

### HSV与RGB的关系
RGB（Red, Green, Blue）模型是基于光的加色混合法则的颜色模型，每个颜色由红、绿、蓝三种基本色的不同强度组合而成。RGB模型主要用于显示设备，如显示器、投影仪等。

转换公式：
- 从RGB到HSV的转换涉及一系列数学计算，主要包括计算最大值、最小值、色差等步骤。
- 从HSV到RGB的转换同样需要一系列计算，主要通过根据色调确定基本颜色分量，然后根据饱和度和亮度调整这些分量。

### 引入HSV的原因
1. 更符合人类视觉感知：HSV模型的设计更符合人类对颜色的直观感受。例如，色调对应于我们对颜色的基本分类（红、绿、蓝等），饱和度对应于颜色的鲜艳程度，亮度对应于颜色的明暗。
2. 颜色分离：在HSV模型中，颜色的属性（色调、饱和度、亮度）是独立的，这使得在图像处理中更容易进行颜色的分离和调整。例如，在图像编辑软件中，用户可以通过调整色调滑块来改变图像的整体颜色，而不会影响其他属性。
3. 图像处理任务：在许多图像处理任务中，HSV模型比RGB模型更为有效。例如，在颜色分割、目标检测等任务中，HSV模型可以帮助更准确地识别和提取特定颜色的对象。
4. 光照鲁棒性：HSV模型中的亮度（Value）属性可以单独调整，这使得在不同光照条件下处理图像时，可以更好地保持颜色的一致性。

### 常见颜色
1. 红色（Red）
   - 色调（Hue）：0° - 10° 或 350° - 360°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

2. 橙色（Orange）
   - 色调（Hue）：10° - 30°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

3. 黄色（Yellow）
   - 色调（Hue）：30° - 60°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

4. 绿色（Green）
   - 色调（Hue）：60° - 180°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

5. 青色（Cyan）
   - 色调（Hue）：180° - 240°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

6. 蓝色（Blue）
   - 色调（Hue）：240° - 300°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

7. 紫色（Purple）
   - 色调（Hue）：300° - 350°
   - 饱和度（Saturation）：70% - 100%
   - 亮度（Value）：50% - 100%

8. 白色（White）
   - 色调（Hue）：任意（通常不考虑）
   - 饱和度（Saturation）：0%
   - 亮度（Value）：100%

9. 黑色（Black）
   - 色调（Hue）：任意（通常不考虑）
   - 饱和度（Saturation）：任意
   - 亮度（Value）：0%

10. 灰色（Gray）
    - 色调（Hue）：任意（通常不考虑）
    - 饱和度（Saturation）：0%
    - 亮度（Value）：0% - 100%

### 基于HSV的像素点识别
```python
def gene_hsv_mask(image_hsv, h_low, s_low, v_low, h_up, s_up, v_up):  # Threshold the image to get only black colors
    if h_low >= h_up:
        return None
    if s_low >= s_up:
        return None
    if v_low >= v_up:
        return None

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_low, s_low, v_low]), np.array([h_up, s_up, v_up]))

    return mask
```

```python
import cv2
import numpy as np

# Define the color threshold for black in HSV color space
black_lower = np.array([0, 0, 0])  # HSV lower bound for black
black_upper = np.array([180, 255, 118])  # HSV upper bound for black

# Define a minimum blob size (area) threshold
min_blob_size = 500  # Adjust this value based on your requirements


def detect_black_blob(image):
    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Threshold the image to get only black colors
    mask = cv2.inRange(hsv, black_lower, black_upper)

    # Perform connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    bboxs = []
    # Rectangle bounding box
    for i in range(1, num_labels):  # Skipping label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter out blobs smaller than the minimum size threshold
        if area < min_blob_size:
            continue

        # Get the bounding box and centroid of each blob
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bboxs.append([x, y, x + w, y + h])

    return bboxs

```

## 级联分类器

Haar级联分类器（Haar Cascade Classifier）是一种基于机器学习的物体检测方法，常用于面部检测和其他物体检测。该算法的核心思想是利用一系列简单的特征和级联分类器来快速识别目标。


## 特征匹配
1. BF(Brute-Force)，暴力特征匹配方法
2. FLANN 最快邻近区特征匹配方法

### 暴力特征匹配
它使用第一组中的每个特征描述子与第二组中的所有特征描述子进行匹配，将最接近的一个匹配返回

交叉验证：两幅图都和另一张图进行匹配，如果某个两个点两次都匹配到了另一张图的对应点，那么是正确的。

[OpenCV官方文档](https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html)
```python
import cv2

img1 = cv2.imread("resize_756_324.png")
img2 = cv2.imread("resize_756_1008.png")

g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 创建sift对象
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
match_result = bf.match(des1, des2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, match_result, None)

cv2.imshow("img3", img3)
cv2.imwrite("result_bf_match.jpg", img3)

cv2.waitKey()
cv2.destroyAllWindows()

```



























