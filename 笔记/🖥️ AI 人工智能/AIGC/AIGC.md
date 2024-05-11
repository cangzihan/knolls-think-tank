---
tags:
  - Stable Diffusion
---

# AIGC

常用的AI绘图模型：
1. [Midjourney](https://midjourney.co/generator): 在线AI绘图网站，免费使用15次。
2. 文心一格: 由百度飞桨、文心大模型的技术创新推出的“AI 作画”产品，在线使用。
3. Stable Diffusion
4. [DALL·E 2](https://openai.com/dall-e-2): 付费订阅ChatGPT后可直接使用

## 项目地址
Stable Diffusion:
https://github.com/CompVis/stable-diffusion

WebUI:
https://github.com/AUTOMATIC1111/stable-diffusion-webui

WebUI插件：
- LoRA: https://github.com/kohya-ss/sd-webui-additional-networks
- ControlNet: https://github.com/Mikubill/sd-webui-controlnet
- IP2P(非ControlNet版): https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix
- AnimateDiff: https://github.com/continue-revolution/sd-webui-animatediff

ComfyUI:
https://github.com/comfyanonymous/ComfyUI/tree/master
```shell
pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple
```
ComfyUI 插件:
- 插件管理器: https://github.com/ltdrdata/ComfyUI-Manager
- lllyasviel/ControlNet:
  1. https://huggingface.co/lllyasviel/Annotators/tree/5bc80eec2b4fddbb743c1e9329e3bd94b2cae14d
  2. https://huggingface.co/dhkim2810/MobileSAM/tree/main

## Stable Diffusion
Stable Diffusion最初是由Heidelberg 大学和[Stability AI](https://stability.ai/), [Runway](https://runwayml.com/)合作的开源项目。

### 版本
#### SD 3
Paper(soon)

combines a [diffusion transformer](https://arxiv.org/abs/2212.09748) architecture and [flow matching](https://arxiv.org/abs/2210.02747).

#### SDXL Turbo
[HuggingFace](https://huggingface.co/stabilityai/sdxl-turbo) | [Paper](https://stability.ai/research/adversarial-diffusion-distillation) (2023.12)
SD Turbo的大号版(高质量)
- 尺寸: 512x512(fix)

#### SD Turbo
[HuggingFace](https://huggingface.co/stabilityai/sd-turbo) (2023.12)
由SD2.1微调而来
- 尺寸: 512x512(fix)

#### SDXL 0.9
[HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) | [Paper](https://arxiv.org/abs/2307.01952)
使用不同尺寸的图像训练（最高1024x1024）

#### SD 2.0/2.1
[SD2.1 HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1) (2.1:2022.12, 2.0: 2022.11)

[SD2.1 Base HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- Hardware: 32 x 8 x A100 GPUs
- 尺寸: 768x768(SD 2.1-v), 512x512(SD 2.1-base)

#### SD 1.5
[Code](https://huggingface.co/runwayml/stable-diffusion-v1-5)(2022.10)

这是SD 1最后一个版本（截止到2024.3），如果你看到什么SD1.8，那肯定是那个人没分清WebUI版本和模型版本。

#### SD 1.1-1.4
[Code](https://github.com/CompVis/stable-diffusion)(2022.8)
- 尺寸: 512x512

### 优化加速
Xformers安装： https://post.smzdm.com/p/axzmd56d/
bash webui.sh --xformers
or
 CUDA_VISIBLE_DEVICES=7 python3 launch.py --listen --enable-insecure-extension-access --xformers

加速效果
OneFlow > [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html) > Aitemplate > Xformers

LLM: https://latent-consistency-models.github.io/

#### 测试
生成图片的大小为`(450,800)`，在使用TensorRT时，生成图片的大小为`(768,450)`

| 环境 | 面部修复 | LoRA | 速度 | +Xformers速度 | +TensorRT速度 |
| --- |:---:|:---:| --- | --- | --- |
| 3060(Notebook) | &#x2713; |   | 3.65it/s | 4.80it/s ||
| 4090 | &#x2713; |   | 7.80it/s | 9.63it/s |32.83it/s|
| 3060(Notebook) | &#x2713;  | &#x2713;  |	3.16it/s | 3.96it/s ||
| 4090 | &#x2713;  | &#x2713;  | 5.54it/s | 6.25it/s |32.87it/s|
| 4090 |  |   | 8.14it/s | 10.06it/s |37.25it/s|

### Embeddings
- ConfyUI:
https://comfyanonymous.github.io/ComfyUI_examples/textual_inversion_embeddings/

To use an embedding put the file in the `models/embeddings` folder then use it in your prompt like I used the SDA768.pt embedding in the previous picture.

Note that you can omit the filename extension so these two are equivalent:

`embedding:SDA768.pt`

`embedding:SDA768`

You can also set the strength of the embedding just like regular words in the prompt:

`(embedding:SDA768:1.2)`

- WebUI
https://zhuanlan.zhihu.com/p/627500143

将embedding文件下载，拷贝至根目录下的embedding目录里

使用：直接在prompt里输入embedding的名字即可，不需要写后缀。新版本的WebUI会自动识别embedding，选择可自动填充prompt

### Controlnet

#### Openpose
这里的Openpose是指借助它提取keypoint特征，而使用[Openpose Editor](https://github.com/fkunn1326/openpose-editor)编辑出来的骨架如果没有输入图像参考，则没有用到Openpose

在 https://huggingface.co/lllyasviel/Annotators/tree/main 中下载3个模型放入`extensions/sd-webui-controlnet/annotator/downloads/openpose`中：

- `body_pose_model.pth`
- `facenet.pth`
- `hand_pose_model.pth`

### 训练

#### LoRA

1. 准备数据: 准备至少10张图像，如果是人，那么背景尽量为白色，不然会被AI学习到背景。放入【文件夹A】

2. 打tag: 使用SD WebUI，点【训练】-【预处理】

   - 其中源目录输入【文件夹A】，创建一个新目录【文件夹B】设为目标目录

   - 自动焦点裁切，使用deepbooru生成说明文字(tags)

   - 设置完后点【输出】

   - 手动修改不正确的标签（在【文件夹B】的`.txt`文件中，可使用[GUI工具](https://github.com/cangzihan/sd_lazy_editor/blob/main/webui.py)）

3. 数据集格式：创建一个新文件夹【文件夹C】，然后在里面再创建一个【文件夹D】命名为"数字_名称"，如“10_face”。其中数字代表训练次数。
然后把【文件夹B】中的所有文件放进去。

4. 训练
```shell
git clone https://github.com/Akegarasu/lora-scripts.git
```

修改`train.ps1`中的内容（代码注释已经很清楚了）

| 变量                  | 说明   |
|---------------------|------|
| `pretrained_model`  |      |
| `train_data_dir`    | 改为【文件夹C】 |
| `max_train_epoches` | 改为14 |
| `output_name`       |      |

然后运行它
```shell
# chmod a+x train.ps1
./train.ps1
```


## 平面设计
[ArchiGAN](https://developer.nvidia.com/blog/archigan-generative-stack-apartment-building-design/?linkId=70968833)

图神经网络方法
[News](https://baijiahao.baidu.com/s?id=1678104857914261902) | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-66823-5_27)(ECCV 2020)

GAN：
https://blog.csdn.net/qq_28941587/article/details/104104823

## Motion

- MDM: [Project](https://guytevet.github.io/mdm-page/) | [Paper](https://arxiv.org/abs/2209.14916) | [Code](https://github.com/GuyTevet/motion-diffusion-model)
- MLD: [Paper](https://arxiv.org/abs/2212.04048) | [Code](https://github.com/ChenFengYe/motion-latent-diffusion)
- T2M-GPT: [Project](https://mael-zys.github.io/T2M-GPT/) | [Paper](https://arxiv.org/abs/2301.06052) | [Code](https://github.com/Mael-zys/T2M-GPT)
- **MoMask**:
[Project](https://ericguo5513.github.io/momask) | [Paper](https://arxiv.org/abs/2312.00063) | [Github](https://github.com/EricGuo5513/momask-codes/tree/main)

Application
- Text-to-motion: is the task of generating motion given an input text prompt.
- Action-to-motion: is the task of generating motion given an input action class, represented by a scalar.
- Motion Editing
  1. Body Part Editing: fix the joints we don’t want to edit and leave the model to generate the rest.
  2. Motion In-Betweening: fix the first and last 25% of the motion, leaving the model to generate the remaining
50% in the middle.

## 3D

Depth Map

3D-GPT: [Project](https://chuny1.github.io/3DGPT/3dgpt.html) | [Paper](https://arxiv.org/abs/2310.12945) | [Code](https://github.com/Chuny1/3DGPT)

DreamScene: [Project](https://dreamscene-project.github.io/) | [Paper](https://arxiv.org/abs/2404.03575) | [Code](https://github.com/DreamScene-Project/DreamScene)

Text2Room: [Project](https://lukashoel.github.io/text-to-room/) | [Code](https://github.com/lukasHoel/text2room)

Text2NeRF: [Project](https://eckertzhang.github.io/Text2NeRF.github.io/) | [Code](https://github.com/eckertzhang/Text2NeRF)

LGM: [Project](https://me.kiui.moe/lgm/) | [Paper](https://arxiv.org/abs/2402.05054) | [Demo](https://huggingface.co/spaces/ashawkey/LGM)

## 动物动作的生成


### MANN

[Video](https://www.youtube.com/watch?v=uFJvRYtjQ4c) | [Paper](https://github.com/sebastianstarke/AI4Animation/blob/master/Media/SIGGRAPH_2018/Paper.pdf) | [Code](https://github.com/sebastianstarke/AI4Animation/tree/master/AI4Animation/SIGGRAPH_2018)

**输入的具体格式**

| 内容 | 成员 | 尺寸 |
| :---: | :---: | :---: |
| state (i)的Trajectory，所有值都**相对于**根节点 | pos.x | 12 |
|| pos.z | 12 |
|| dir.x | 12 |
|| dir.z | 12 |
|| vel.x | 12 |
|| vel.z | 12 |
|| speed | 12 |
|| Styles的6维one-hot向量| 12 |
| state (i-1) 的关节 | pos.x | 27 |
|| pos.y | 27 |
|| pos.z | 27 |
|| foward.x | 27 |
|| foward.y | 27 | | 27 |
|| foward.z | 27 |
|| up.x | 27 |
|| up.y | 27 |
|| up.z | 27 |
|| vel.x | 27 |
|| vel.y | 27 |
|| vel.z | 27 |

**输出的具体格式**

| 内容 | 成员 | 尺寸 |
| :---: | :---: | :---: |
| state (i+1)的Trajectory，所有值都**相对于**根节点 | pos.x | 6 |
|| pos.z | 6 |
|| dir.x | 6 |
|| dir.z | 6 |
|| vel.x | 6 |
|| vel.z | 6 |
| state (i-1) 的关节 | pos.x | 27 |
|| pos.y | 27 |
|| pos.z | 27 |
|| foward.x | 27 |
|| foward.y | 27 | | 27 |
|| foward.z | 27 |
|| up.x | 27 |
|| up.y | 27 |
|| up.z | 27 |
|| vel.x | 27 |
|| vel.y | 27 |
|| vel.z | 27 |
| root节点相对于上一帧的位移 | (x, 角度, z) | 3 |

MoE 指的是 Mixture of Experts（专家混合模型），是一种用于构建深度神经网络的架构。这种架构通常包括两个关键组件：专家网络和门控网络。

1. 专家网络（Experts）： 这是多个神经网络模块的集合，每个模块被称为一个专家。每个专家被设计为在处理输入数据的特定方面上表现出色。例如，对于图像分类任务，不同的专家可能擅长识别不同类别的物体。

2. 门控网络（Gating Network）： 门控网络用于确定在给定输入上哪个专家应该发挥作用。门控网络输出一组权重，这些权重表示每个专家对给定输入的贡献。这些权重通常是在0到1之间的值，它们的和等于1。

整个 MoE 模型的输出是所有专家的输出的加权和，权重由门控网络确定。这使得 MoE 能够在不同的输入情况下动态地选择不同的专家来执行任务。

MoE 的优点之一是其能够处理复杂的、多模态的数据分布，因为不同的专家可以专注于处理不同方面的数据。这种结构也有助于提高模型的容量和表达能力，使其能够更好地适应复杂的任务。 MoE 结构常常在涉及大规模神经网络和复杂任务的情况下取得了良好的性能。

权重文件命名规则：

1. `cp[0-2]_[a,b][0-7].bin`，一共3x2x8=48个文件
  - `ExpertWeights.py`
  - a,b 表示 $\alpha$和$\beta$
  - 0-7 表示专家的索引

2. `wc[0-2]_[b,w].bin`，一共3x2=6个文件
  - `Gating.py`
  - w 表示weight
  - b 表示bias

3. `[X,Y][mean,std].bin`，一共2x2=4个文件
  - `Utils.py`
  - mean 表示均值
  - std 表示方差

| 变量 | 尺寸 |
| :---: | :---: |
| Xmean | 480 x 1 |
| Xstd | 480 x 1 |
| Ymean | 363 x 1 |
| Ystd | 363 x 1 |
| wc0_w | 32 x 19 |
| wc0_b | 32 x 1 |
| wc1_w | 32 x 32 |
| wc1_b | 32 x 1 |
| wc2_w | 8 x 32 |
| wc2_b | 8 x 1 |
| cp0_a0-7 | 512 x 480 |
| cp0_b0-7 | 512 x 1 |
| cp1_a0-7 | 512 x 512 |
| cp1_b0-7 | 512 x 1 |
| cp2_a0-7 | 363 x 512 |
| cp2_b0-7 | 363 x 1 |


## 其他
**Custom Diffusion**
[Home](https://www.cs.cmu.edu/~custom-diffusion/results.html) |
[Github](https://github.com/adobe-research/custom-diffusion)

https://zhuanlan.zhihu.com/p/620852185


**数字人**

Wav2lip：https://github.com/Rudrabha/Wav2Lip

EasyWav2lip: https://github.com/anothermartz/Easy-Wav2Lip

facefusion2.5: https://github.com/facefusion/facefusion

SadTalker-Video-Lip-Sync: https://github.com/Zz-ww/SadTalker-Video-Lip-Sync

**换脸**

IPAdapter （通常会伴随其他元素替换）

ReActor

facefusion

DeepFaceLive




## 名词解释
- **DreamBooth**
is a training technique that updates the entire diffusion model by training on just a few images of a subject or style. It works by associating a special word in the prompt with the example images.

- **LoRA**
(Low-Rank Adaptation of Large Language Models) is a popular and lightweight training technique that significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained.

- **SD[1.4/1.5/2.0]**
Stable Diffusion.

- **SVD**
Stable Video Diffusion.
