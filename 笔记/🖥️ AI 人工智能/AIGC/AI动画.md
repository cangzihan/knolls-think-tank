---
tags:
  - AIGC
  - Text-to-Video
  - Image-to-Video
---

# AI动画
- Gen-1(2023): Runway 公司的Video to Video的AI产品。https://arxiv.org/abs/2302.03011
- Gen-2(2023): Runway 公司的Text/Image to Video的AI产品。  [Paper Coming Soon](https://research.runwayml.com/gen2)
- Animate Diff: [open source](https://github.com/guoyww/AnimateDiff/), 结合Stable Diffusion的开源视频生成模型。 需要12G显存）https://arxiv.org/abs/2307.04725
- Deforum: open source, 使用 Stable Diffusion 创建动画视频的工具。
- Kaiber.ai: AI创作者平台，输入图片/视频/prompt生成视频，Audioreactivity。
- Pika Labs: Text-to-Video，Discord上通过聊天机器人使用

## Text-to-Video

### Gen-2

给一个图片和一段Prompt生成视频，也可以只给文字。

[体验入口](http://app.runwayml.com/)

### Pika Labs

### AnimateDiff
#### AnimateDiff & ControlNet & Pose
![img](assets/animatediff_pose_workflow.png)
将AnimateDiff、动作序列和ControlNet并入一个工作流，实现文生视频（和Animate Anyone相比，效果稍微差一些，但Animate Anyone是图生视频）
![img](assets/268628360-bf926f52-da97-4fb4-b86a-8b26ef5fab04.gif)
![img](assets/AnimateDiff_00037_.gif)

#### AnimateDiff + 动态LoRA实现视角控制
将AnimateDiff在原有算法的基础上结合视角控制的LoRA并入一个工作流
![img](assets/animatediff_dlora_workflow.png)

### 文生图 + SVD
![img](assets/T2I_SVD.png)
*通过文字生成视频序列工作流*

### Sora
Sora is a diffusion model, which generates a video by starting off with one that looks like static noise and gradually transforms it by removing the noise over many steps.

Similar to GPT models, Sora uses a transformer architecture, unlocking superior scaling performance.

Sora builds on past research in DALL·E and GPT models. It uses the recaptioning technique from DALL·E 3, which involves generating highly descriptive captions for the visual training data.

Sora is a diffusion model; given input noisy patches (and conditioning information like text prompts), it’s trained to predict the original “clean” patches. Importantly, Sora is a diffusion transformer.
![img](assets/figure-diffusion.avif)

高斯模型转换，将生成的视频分解成29个图片一个文件夹的形式，使用[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)库训练。

Sora的一个复现项目 [Project(已失效)](https://pku-yuangroup.github.io/Open-Sora-Plan/blog_cn.html) | [Code](https://github.com/PKU-YuanGroup/Open-Sora-Plan) | [Demo](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0) | [Report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md)

## Image-to-Video
可以分为两类，一类是给图片(可附加文字)生成视频，另一类是给图片和对应的特征(语义，关节点，normal map等)生成视频，如果每一帧都能提供特征，理论上可以使视频的长度很长，不受限制。

### Gen-2

给一个图片和一段Prompt生成视频，也可以只给图片，如果不想视频和图片差异太大就只给图片。

[体验入口](http://app.runwayml.com/)

### Animate Anyone
[Project](https://humanaigc.github.io/animate-anyone/) | [Code](https://github.com/HumanAIGC/AnimateAnyone/tree/main)

Animate Anyone 是阿里提出的用于人物动画场景的图生视频的方法，只需一张人物照片，再配合骨骼动画引导，就能生成动画视频。

### Magic Animate

- diffusion model

[Project](https://showlab.github.io/magicanimate/) | [Code](https://github.com/magic-research/magic-animate)

新加坡国立大学与字节合作推出MagicAnimate框架，可以通过一张照片和一组语义分割图像生成动画视频。

### Stable Video Diffusion
![img](assets/SVD_workflow.png)
*通过任意一张图片生成视频序列工作流*

![img](assets/Rocket-To-The-Moon-SpaceX8217s-Starship-Takes-Flight-Once-Again_6509d8fb577ef.jpg)
*原始图片*

![img](assets/SVD_00039.gif)
*SVD生成视频*

![img](assets/SVD_00039_2.gif)
*加入补帧算法后效果*

### Sora

## Video-to-Video
### Gen-1
[体验入口](http://app.runwayml.com/)

### Sora
主要功能有：
- 修改风格和环境
- **Connecting videos:** 输入两个video，生成一个新视频包含有两个视频的内容和过渡
