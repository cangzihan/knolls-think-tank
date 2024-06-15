---
tags:
  - 3D
  - 计算机图形学
  - 神经网络
---

# NeRF
[Paper](https://arxiv.org/abs/2003.08934)

## 原理
输入$(x,y,z,\theta, \phi)$，输出采样点颜色和不透明度$(R,G,B,\alpha)$

输入：
- Position(相机位置为起点的采样点位置): $(x,y,z)$
- Direction(观测角度): $(\theta, \phi)$这个是[球坐标](https://zh.wikipedia.org/zh-cn/%E7%90%83%E5%BA%A7%E6%A8%99%E7%B3%BB)的表现方法

映射：
所以一个深度网络模型只能训练一个物体，这个三维模型的**渲染信息**就隐式地储存在深度学习模型中，而不是像点云体素是显示的表示。
一个神经网络模型只需要40mb的大小就能储存下3D模型内容。

输出：
- 像素点对应射线上的**一组采样点**的颜色值$(R, G, B)$和不透明度$\alpha$
- 射线方向上对采样渲染点进行积分，在第一次出现波峰对该像素点的着色影响最大

## CityDreamer

[Code](https://github.com/hzxie/CityDreamer) | [Paper](https://arxiv.org/abs/2309.00610)
