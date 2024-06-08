---
tags:
  - Stable Diffusion
  - AI绘图
---

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

### 原理

<div class="theme-image">
  <img src="./assets/SD.png" alt="Light Mode Image" class="light-mode">
  <img src="./assets/dark_SD.png" alt="Dark Mode Image" class="dark-mode">
</div>

#### 分词器(tokenizer)
text先由CLIP进行标记化，CLIP是由OpenAI开发（英文版）的一种多模态模型，旨在理解图像和文本之间的关系。CLIP的训练过程包括：
1. **数据集**：CLIP使用包含图像和相应描述性文本对的大规模数据集进行训练。
2. **对比学习**：CLIP采用对比学习的方法，通过最大化图像和对应文本的相似性（而不是与随机文本的相似性）来训练模型。CLIP使用了两个独立的神经网络，一个用于处理图像（图像编码器），一个用于处理文本（文本编码器）。

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)


功能：
- 图像-文本相似性评估：CLIP可以计算任意图像和文本之间的相似性，找到最相关的图像或文本。
- 零样本分类：通过对文本描述进行编码，CLIP可以在没有明确训练过的分类任务上进行图像分类。
- 图像生成指导：在生成任务中，CLIP可以提供目标图像的特征指导，帮助生成高质量的图像。

#### 令牌化(Tokenization)
max 75个令牌

#### 嵌入/标签(Embedding)
ViT-L/14

#### VAE（Variational Autoencoder）
VAE是一种生成模型，用于学习数据的潜在表示并生成新数据。VAE包括两个主要部分：
1. 编码器（Encoder）：将输入数据（例如图像）编码为潜在表示（通常是一个潜在向量）。
2. 解码器（Decoder）：从潜在表示中重建输入数据。
VAE的训练目标是最大化变分下界（Variational Lower Bound），以使重建的图像尽可能接近原始图像，并使潜在表示的分布接近先验分布（通常是标准正态分布）。

#### why VAE? not CLIP
CLIP和VAE的区别

CLIP（Contrastive Language-Image Pre-Training）：

- 功能：CLIP是一个用于理解图像和文本之间关系的多模态模型。它包含两个部分：图像编码器和文本编码器。
- 作用：CLIP用于计算图像和文本之间的相似性。通过对比学习，CLIP能够将图像和文本映射到同一个特征空间中，从而可以进行相似性评估。
限制：CLIP没有解码器部分，因此无法直接生成图像。它主要用于评估和指导生成过程，而不是直接参与图像生成。

VAE（Variational Autoencoder）：

功能：VAE是一种生成模型，用于学习数据的潜在表示，并能够从潜在表示生成新数据。VAE包含编码器和解码器两部分。
作用：在Stable Diffusion中，VAE用于将图像编码为潜在向量（编码器），并从潜在向量生成图像（解码器）。
优势：VAE的解码器部分在图像生成过程中起关键作用，能够从扩散模型生成的潜在表示重建图像。

为什么使用VAE而不是CLIP进行解码

CLIP没有解码器部分，所以它不能直接从潜在表示生成图像。CLIP的主要作用是提供文本和图像之间的相似性指导。例如，在生成过程中，CLIP可以帮助确保生成的图像与给定的文本描述相符，但实际的图像生成和解码过程需要依赖其他模型（如VAE）。

总结

在Stable Diffusion中，VAE的解码器用于将扩散模型生成的潜在表示转化为图像。CLIP则用于提供文本-图像相似性指导，确保生成的图像符合文本描述。由于CLIP缺乏解码器部分，Stable Diffusion使用VAE来完成图像的实际生成。

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

### Comfy UI

#### 共享路径设置
个人习惯将模型路径设定为一个统一的路径，使任何平台的WebUI和ComfyUI都用同一路径下的模型，节省空间。

- Windows版: 修改`ComgyUI/folder_paths.py`
```python
import os
import time
import logging

supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl'])

folder_names_and_paths = {}

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(base_path, "models")
share_path = "C:\Software\ShareModel" # [!code ++]
folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions) # [!code --]
folder_names_and_paths["checkpoints"] = ([os.path.join(share_path, "stablediffusion")], supported_pt_extensions) # [!code ++]
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions) # [!code --]
folder_names_and_paths["loras"] = ([os.path.join(share_path, "loras")], supported_pt_extensions) # [!code ++]
folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
folder_names_and_paths["clip"] = ([os.path.join(models_dir, "clip")], supported_pt_extensions)
folder_names_and_paths["unet"] = ([os.path.join(models_dir, "unet")], supported_pt_extensions)
folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], ["folder"])
folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)

#folder_names_and_paths["controlnet"] = ([os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
folder_names_and_paths["controlnet"] = ([os.path.join(share_path, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
```

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

## SD & 3D Model

### 贴图生成
#### 无缝贴图
教程: https://www.bilibili.com/video/BV1Kp42117Mv
::: code-group
```json [无缝贴图-AI生成v3.1]
{
  "last_node_id": 329, "last_link_id": 695,
  "nodes": [
    {
      "id": 172, "type": "PrimitiveNode", "pos": [1468.0768300781262, 350], "size": {"0": 210, "1": 80}, "flags": {}, "order": 0, "mode": 0,
      "outputs": [
        {"name": "INT", "type": "INT", "links": [282, 289], "slot_index": 0, "widget": {"name": "x_offset"}, "label": "INT"}
      ],
      "properties": {"Run widget replace on values": false}, "widgets_values": [1020, "fixed"]
    },
    {"id": 227, "type": "Reroute", "pos": [1538.0768300781262, 500], "size": [75, 26], "flags": {}, "order": 29, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 695, "label": ""}], "outputs": [{"name": "", "type": "IMAGE", "links": [434, 435, 436, 437], "slot_index": 0, "label": ""}], "properties": {"showOutputText": false, "horizontal": false}},
    {"id": 6, "type": "CLIPTextEncode", "pos": [630, 210], "size": {"0": 210, "1": 90}, "flags": {"collapsed": true}, "order": 18, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 224, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 98, "widget": {"name": "text"}, "label": "\u6587\u672c"}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [48], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["1girl"], "color": "#232", "bgcolor": "#353"},
    {"id": 7, "type": "CLIPTextEncode", "pos": [640, 260], "size": {"0": 210, "1": 100}, "flags": {"collapsed": true}, "order": 21, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 225, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 100, "widget": {"name": "text"}, "label": "\u6587\u672c", "slot_index": 1}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["text, watermark"], "color": "#322", "bgcolor": "#533"},
    {"id": 245, "type": "CLIPTextEncode", "pos": [310, 1110], "size": {"0": 210, "1": 90}, "flags": {"collapsed": true}, "order": 20, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 457, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 464, "widget": {"name": "text"}, "label": "\u6587\u672c", "slot_index": 1}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [459], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["1girl"], "color": "#232", "bgcolor": "#353"},
    {"id": 3, "type": "KSampler", "pos": [810, 220], "size": {"0": 260, "1": 470}, "flags": {}, "order": 25, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 223, "label": "\u6a21\u578b"}, {"name": "positive", "type": "CONDITIONING", "link": 48, "label": "\u6b63\u9762\u6761\u4ef6"}, {"name": "negative", "type": "CONDITIONING", "link": 6, "label": "\u8d1f\u9762\u6761\u4ef6"}, {"name": "latent_image", "type": "LATENT", "link": 431, "label": "Latent"}, {"name": "seed", "type": "INT", "link": 506, "widget": {"name": "seed"}, "label": "\u968f\u673a\u79cd", "slot_index": 4}, {"name": "steps", "type": "INT", "link": 503, "widget": {"name": "steps"}, "label": "\u6b65\u6570"}, {"name": "cfg", "type": "FLOAT", "link": 502, "widget": {"name": "cfg"}, "label": "CFG"}, {"name": "sampler_name", "type": "COMBO", "link": 504, "widget": {"name": "sampler_name"}, "label": "\u91c7\u6837\u5668", "slot_index": 7}, {"name": "scheduler", "type": "COMBO", "link": 505, "widget": {"name": "scheduler"}, "label": "\u8c03\u5ea6\u5668", "slot_index": 8}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7], "slot_index": 0, "label": "Latent"}], "properties": {"Node name for S&R": "KSampler"}, "widgets_values": [386213472092672, "randomize", 8, 1, "euler", "sgm_uniform", 1]}, {"id": 65, "type": "SDXLPromptStyler", "pos": [560, 330], "size": {"0": 230, "1": 170}, "flags": {"collapsed": false}, "order": 15, "mode": 0, "inputs": [{"name": "text_positive", "type": "STRING", "link": 204, "widget": {"name": "text_positive"}, "label": "\u6b63\u9762\u6761\u4ef6"}, {"name": "text_negative", "type": "STRING", "link": 205, "widget": {"name": "text_negative"}, "label": "\u8d1f\u9762\u6761\u4ef6"}], "outputs": [{"name": "positive_prompt_text_g", "type": "STRING", "links": [98, 134, 464], "shape": 3, "label": "positive_prompt_text_g", "slot_index": 0}, {"name": "negative_prompt_text_g", "type": "STRING", "links": [100, 135, 465], "shape": 3, "label": "negative_prompt_text_g", "slot_index": 1}], "properties": {"Node name for S&R": "SDXLPromptStyler"}, "widgets_values": ["1 girl, long hair, dress, 3/4 profile, ", "text, watermark", "sai-texture", true, true, true], "color": "#232", "bgcolor": "#353"},
    {"id": 211, "type": "PreviewImage", "pos": [1988.0768300781262, 250], "size": {"0": 210, "1": 250}, "flags": {}, "order": 33, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 361, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 268, "type": "PrimitiveNode", "pos": [570, 820], "size": {"0": 210, "1": 80}, "flags": {}, "order": 1, "mode": 0, "outputs": [{"name": "FLOAT", "type": "FLOAT", "links": [497, 502], "slot_index": 0, "widget": {"name": "cfg"}, "label": "FLOAT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [1, "fixed"]}, {"id": 274, "type": "PrimitiveNode", "pos": [570, 680], "size": {"0": 210, "1": 80}, "flags": {}, "order": 2, "mode": 0, "outputs": [{"name": "INT", "type": "INT", "links": [501, 503], "slot_index": 0, "widget": {"name": "steps"}, "label": "INT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [8, "fixed"]}, {"id": 91, "type": "CheckpointLoader|pysssss", "pos": [230, 230], "size": {"0": 210, "1": 122}, "flags": {}, "order": 3, "mode": 0, "outputs": [{"name": "MODEL", "type": "MODEL", "links": [223, 454], "shape": 3, "label": "\u6a21\u578b", "slot_index": 0}, {"name": "CLIP", "type": "CLIP", "links": [224, 225, 457, 458], "shape": 3, "label": "CLIP", "slot_index": 1}, {"name": "VAE", "type": "VAE", "links": [609], "shape": 3, "label": "VAE", "slot_index": 2}], "properties": {"Node name for S&R": "CheckpointLoader|pysssss"}, "widgets_values": [{"content": "SDXL-lightning/sdxl_lightning_8step.safetensors", "image": null}, "[none]"]}, {"id": 306, "type": "Reroute", "pos": [1140, 150], "size": [75, 26], "flags": {}, "order": 13, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 609}], "outputs": [{"name": "", "type": "VAE", "links": [613, 617], "slot_index": 0}], "properties": {"showOutputText": false, "horizontal": false}}, {"id": 176, "type": "PreviewImage", "pos": [1988.0768300781262, 630], "size": {"0": 200, "1": 250}, "flags": {}, "order": 31, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 290, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 181, "type": "Image Overlay", "pos": [1748.0768300781262, 630], "size": {"0": 210, "1": 290}, "flags": {}, "order": 30, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 693, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 434, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "y_offset", "type": "INT", "link": 289, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb", "slot_index": 4}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [290, 405], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 0, 1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1},
    {"id": 264, "type": "PreviewImage", "pos": [1090, 370], "size": {"0": 290, "1": 320}, "flags": {}, "order": 28, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 489, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 270, "type": "PrimitiveNode", "pos": [810, 810], "size": {"0": 210, "1": 110}, "flags": {}, "order": 4, "mode": 0, "outputs": [{"name": "COMBO", "type": "COMBO", "links": [498, 504], "slot_index": 0, "widget": {"name": "sampler_name"}, "label": "COMBO"}], "properties": {"Run widget replace on values": false}, "widgets_values": ["euler", "fixed", ""]},
    {"id": 271, "type": "PrimitiveNode", "pos": [1060, 810], "size": {"0": 210, "1": 110}, "flags": {}, "order": 5, "mode": 0, "outputs": [{"name": "COMBO", "type": "COMBO", "links": [499, 505], "slot_index": 0, "widget": {"name": "scheduler"}, "label": "COMBO"}], "properties": {"Run widget replace on values": false}, "widgets_values": ["sgm_uniform", "fixed", ""]}, {"id": 307, "type": "Reroute", "pos": [109.90929193787348, 1038.4640011941774], "size": [75, 26], "flags": {}, "order": 16, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 613}], "outputs": [{"name": "", "type": "VAE", "links": [614, 615], "slot_index": 0}], "properties": {"showOutputText": false, "horizontal": false}}, {"id": 290, "type": "SaveImage", "pos": [2414.773395543872, 1209.6649475057209], "size": {"0": 380, "1": 480}, "flags": {}, "order": 53, "mode": 2, "inputs": [{"name": "images", "type": "IMAGE", "link": 659, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["normal-map"]},
    {"id": 291, "type": "SaveImage", "pos": [2840.6181026141826, 1216.2230412752522], "size": {"0": 400, "1": 480}, "flags": {}, "order": 54, "mode": 2, "inputs": [{"name": "images", "type": "IMAGE", "link": 660, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["bump-map"]}, {"id": 288, "type": "AIO_Preprocessor", "pos": [2840.6181026141826, 1076.2230412752522], "size": {"0": 310, "1": 82}, "flags": {}, "order": 51, "mode": 2, "inputs": [{"name": "image", "type": "IMAGE", "link": 691, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [660], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "AIO_Preprocessor"}, "widgets_values": ["MiDaS-DepthMapPreprocessor", 1024]},
    {"id": 286, "type": "AIO_Preprocessor", "pos": [2428.2527168593847, 1080.2547984062505], "size": {"0": 310, "1": 82}, "flags": {}, "order": 50, "mode": 2, "inputs": [{"name": "image", "type": "IMAGE", "link": 690, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [659], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "AIO_Preprocessor"}, "widgets_values": ["BAE-NormalMapPreprocessor", 1024]}, {"id": 179, "type": "Image Overlay", "pos": [1750, 220], "size": {"0": 210, "1": 290}, "flags": {}, "order": 32, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 405, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 435, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 282, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb", "slot_index": 3}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [361, 669], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 1020, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 222, "type": "Image Overlay", "pos": [2230, 220], "size": {"0": 210, "1": 290}, "flags": {}, "order": 34, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 669, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 437, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [422, 670], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 1020, 1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 220, "type": "Image Overlay", "pos": [2230, 620], "size": {"0": 210, "1": 290}, "flags": {}, "order": 36, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 670, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 436, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [416, 671], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 0, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 223, "type": "PreviewImage", "pos": [2460, 240], "size": {"0": 210, "1": 250}, "flags": {}, "order": 35, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 422, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 221, "type": "PreviewImage", "pos": [2460, 620], "size": {"0": 210, "1": 250}, "flags": {}, "order": 37, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 416, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}},
    {"id": 317, "type": "Reroute", "pos": [1133.988006132812, 1061.076388007812], "size": [75, 26], "flags": {}, "order": 43, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 687, "label": ""}], "outputs": [{"name": "", "type": "IMAGE", "links": [633, 634, 637, 643], "slot_index": 0}], "properties": {"showOutputText": false, "horizontal": false}}, {"id": 324, "type": "ImageToMask", "pos": [1113.988006132812, 1631.076388007812], "size": {"0": 210, "1": 60}, "flags": {}, "order": 26, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 677, "label": "\u56fe\u50cf"}], "outputs": [{"name": "MASK", "type": "MASK", "links": [683, 684, 685], "shape": 3, "label": "\u906e\u7f69", "slot_index": 0}], "properties": {"Node name for S&R": "ImageToMask"}, "widgets_values": ["red"]}, {"id": 313, "type": "Image Overlay", "pos": [1383.988006132812, 1441.076388007812], "size": {"0": 210, "1": 290}, "flags": {}, "order": 45, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 636, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 637, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 684, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "y_offset", "type": "INT", "link": 638, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [642], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 1280, 256, 0, -1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1},
    {"id": 318, "type": "Image Overlay", "pos": [1623.9880061328115, 1311.076388007812], "size": {"0": 210, "1": 290}, "flags": {}, "order": 46, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 642, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 643, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 685, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 644, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb", "slot_index": 3}, {"name": "y_offset", "type": "INT", "link": 645, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb", "slot_index": 4}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [639], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 256, 256, -1020, -1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 315, "type": "SaveImage", "pos": [1853.9880061328115, 1261.076388007812], "size": {"0": 460, "1": 480}, "flags": {}, "order": 49, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 689, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["color-map"]}, {"id": 322, "type": "ImageScale", "pos": [1113.988006132812, 1271.076388007812], "size": {"0": 210, "1": 130}, "flags": {}, "order": 17, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 676, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [675], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageScale"}, "widgets_values": ["bilinear", 1590, 1590, "disabled"]}, {"id": 323, "type": "ImageCrop", "pos": [1113.988006132812, 1451.076388007812], "size": {"0": 210, "1": 130}, "flags": {}, "order": 24, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 675, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [677], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageCrop"}, "widgets_values": [1276, 1276, 0, 0]}, {"id": 129, "type": "VAEEncode", "pos": [110, 1110], "size": {"0": 140, "1": 46}, "flags": {}, "order": 38, "mode": 0, "inputs": [{"name": "pixels", "type": "IMAGE", "link": 671, "label": "\u56fe\u50cf"}, {"name": "vae", "type": "VAE", "link": 614, "label": "VAE", "slot_index": 1}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [208], "shape": 3, "label": "Latent", "slot_index": 0}], "properties": {"Node name for S&R": "VAEEncode"}},
    {"id": 249, "type": "KSampler", "pos": [400, 1220], "size": {"0": 260, "1": 470}, "flags": {}, "order": 40, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 454, "label": "\u6a21\u578b"}, {"name": "positive", "type": "CONDITIONING", "link": 459, "label": "\u6b63\u9762\u6761\u4ef6", "slot_index": 1}, {"name": "negative", "type": "CONDITIONING", "link": 460, "label": "\u8d1f\u9762\u6761\u4ef6", "slot_index": 2}, {"name": "latent_image", "type": "LATENT", "link": 466, "label": "Latent", "slot_index": 3}, {"name": "cfg", "type": "FLOAT", "link": 497, "widget": {"name": "cfg"}, "label": "CFG"}, {"name": "sampler_name", "type": "COMBO", "link": 498, "widget": {"name": "sampler_name"}, "label": "\u91c7\u6837\u5668"}, {"name": "scheduler", "type": "COMBO", "link": 499, "widget": {"name": "scheduler"}, "label": "\u8c03\u5ea6\u5668"}, {"name": "seed", "type": "INT", "link": 500, "widget": {"name": "seed"}, "label": "\u968f\u673a\u79cd"}, {"name": "steps", "type": "INT", "link": 501, "widget": {"name": "steps"}, "label": "\u6b65\u6570"}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [461], "slot_index": 0, "label": "Latent"}], "properties": {"Node name for S&R": "KSampler"}, "widgets_values": [1065189257968362, "randomize", 8, 1, "euler", "sgm_uniform", 1]},
    {"id": 132, "type": "SetLatentNoiseMask", "pos": [110, 1220], "size": {"0": 176.39999389648438, "1": 46}, "flags": {}, "order": 39, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 208, "label": "Latent"}, {"name": "mask", "type": "MASK", "link": 491, "label": "\u906e\u7f69", "slot_index": 1}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [466], "shape": 3, "label": "Latent", "slot_index": 0}], "properties": {"Node name for S&R": "SetLatentNoiseMask"}}, {"id": 246, "type": "CLIPTextEncode", "pos": [310, 1160], "size": {"0": 210, "1": 100}, "flags": {"collapsed": true}, "order": 23, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 458, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 465, "widget": {"name": "text"}, "label": "\u6587\u672c", "slot_index": 1}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [460], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["text, watermark"], "color": "#322", "bgcolor": "#533"}, {"id": 263, "type": "PreviewImage", "pos": [690, 1230], "size": {"0": 310, "1": 350}, "flags": {}, "order": 42, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 487, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 312, "type": "Image Overlay", "pos": [1384.784878124999, 1092.8564836914059], "size": {"0": 210, "1": 290}, "flags": {}, "order": 44, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 633, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 634, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 683, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 635, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [636], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 256, 1280, -1020, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 316, "type": "PrimitiveNode", "pos": [1113.988006132812, 1141.076388007812], "size": {"0": 210, "1": 80}, "flags": {}, "order": 6, "mode": 0, "outputs": [{"name": "INT", "type": "INT", "links": [635, 638, 644, 645], "slot_index": 0, "widget": {"name": "x_offset"}, "label": "INT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [-1020, "fixed"]}, {"id": 206, "type": "LoadImageMask", "pos": [110, 1340], "size": {"0": 250, "1": 320}, "flags": {}, "order": 7, "mode": 0, "outputs": [{"name": "MASK", "type": "MASK", "links": [491, 493], "shape": 3, "label": "\u906e\u7f69", "slot_index": 0}], "properties": {"Node name for S&R": "LoadImageMask"}, "widgets_values": ["alpha \u8d34\u56fe.png", "alpha", "image"]}, {"id": 266, "type": "MaskToImage", "pos": [850, 1150], "size": {"0": 140, "1": 30}, "flags": {}, "order": 14, "mode": 0, "inputs": [{"name": "mask", "type": "MASK", "link": 493, "label": "\u906e\u7f69"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [676], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "MaskToImage"}}, {"id": 248, "type": "VAEDecode", "pos": [680, 1120], "size": {"0": 140, "1": 50}, "flags": {}, "order": 41, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 461, "label": "Latent"}, {"name": "vae", "type": "VAE", "link": 615, "label": "VAE", "slot_index": 1}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [487, 687], "slot_index": 0, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "VAEDecode"}}, {"id": 314, "type": "ImageCrop", "pos": [1613.9880061328115, 1081.076388007812], "size": {"0": 210, "1": 130}, "flags": {}, "order": 47, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 639, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [688], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageCrop"}, "widgets_values": [1020, 1020, 0, 0]}, {"id": 319, "type": "PlaySound|pysssss", "pos": [2090, 1090], "size": {"0": 210, "1": 110}, "flags": {}, "order": 52, "mode": 0, "inputs": [{"name": "any", "type": "*", "link": 692, "label": "\u8f93\u5165"}], "outputs": [{"name": "*", "type": "*", "links": null, "shape": 6}], "properties": {"Node name for S&R": "PlaySound|pysssss"}, "widgets_values": ["always", 1, "notify.mp3"]}, {"id": 327, "type": "ImageScale", "pos": [1854.7848781249986, 1082.8564836914059], "size": {"0": 210, "1": 130}, "flags": {}, "order": 48, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 688, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [689, 690, 691, 692], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageScale"}, "widgets_values": ["bilinear", 1024, 1024, "disabled"]},
    {"id": 328, "type": "EmptyImage", "pos": [1480, 710], "size": [210, 130], "flags": {}, "order": 8, "mode": 0, "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [693], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "EmptyImage"}, "widgets_values": [1276, 1276, 1, 999999]}, {"id": 8, "type": "VAEDecode", "pos": [1110, 240], "size": {"0": 140, "1": 50}, "flags": {}, "order": 27, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 7, "label": "Latent"}, {"name": "vae", "type": "VAE", "link": 617, "label": "VAE", "slot_index": 1}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [489, 695], "slot_index": 0, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "VAEDecode"}}, {"id": 126, "type": "DeepTranslatorTextNode", "pos": [90, 400], "size": [230, 250], "flags": {}, "order": 9, "mode": 0, "outputs": [{"name": "text", "type": "STRING", "links": [204], "shape": 3, "label": "\u6587\u672c", "slot_index": 0}], "properties": {"Node name for S&R": "DeepTranslatorTextNode"}, "widgets_values": ["auto", "english", "disable", "", "", "GoogleTranslator [free]", "rock", "proxy_hide", "authorization_hide"], "color": "#232", "bgcolor": "#353"}, {"id": 127, "type": "DeepTranslatorTextNode", "pos": [90, 690], "size": [230, 250], "flags": {}, "order": 10, "mode": 0, "outputs": [{"name": "text", "type": "STRING", "links": [205], "shape": 3, "label": "\u6587\u672c", "slot_index": 0}], "properties": {"Node name for S&R": "DeepTranslatorTextNode"}, "widgets_values": ["auto", "english", "disable", "", "", "GoogleTranslator [free]", "", "proxy_hide", "authorization_hide"], "color": "#322", "bgcolor": "#533"},
    {"id": 92, "type": "ShowText|pysssss", "pos": [330, 410], "size": [220, 160], "flags": {"collapsed": false}, "order": 19, "mode": 0, "inputs": [{"name": "text", "type": "STRING", "link": 134, "widget": {"name": "text"}, "label": "\u6587\u672c"}], "outputs": [{"name": "STRING", "type": "STRING", "links": null, "shape": 6, "label": "\u5b57\u7b26\u4e32"}], "properties": {"Node name for S&R": "ShowText|pysssss"}, "widgets_values": [["ethereal fantasy concept art of  psychedelic style 1 girl, \u9648\u5c0f\u7ead, long hair, dress, looking back,   . vibrant colors, swirling patterns, abstract forms, surreal, trippy . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"], "ethereal fantasy concept art of  psychedelic style 1 girl, \u9648\u5c0f\u7ead, long hair, dress, looking back,   . vibrant colors, swirling patterns, abstract forms, surreal, trippy . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy", "texture rock top down close-up"], "color": "#232", "bgcolor": "#353"},
    {"id": 272, "type": "PrimitiveNode", "pos": [570, 550], "size": {"0": 210, "1": 80}, "flags": {}, "order": 11, "mode": 0, "outputs": [{"name": "INT", "type": "INT", "links": [500, 506], "slot_index": 0, "widget": {"name": "seed"}, "label": "INT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [386213472092672, "randomize"]},
    {"id": 93, "type": "ShowText|pysssss", "pos": [330, 610], "size": [220, 130], "flags": {"collapsed": false}, "order": 22, "mode": 0, "inputs": [{"name": "text", "type": "STRING", "link": 135, "widget": {"name": "text"}, "label": "\u6587\u672c"}], "outputs": [{"name": "STRING", "type": "STRING", "links": null, "shape": 6, "label": "\u5b57\u7b26\u4e32"}], "properties": {"Node name for S&R": "ShowText|pysssss"}, "widgets_values": [["photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, monochrome, black and white, low contrast, realistic, photorealistic, plain, simple, text, watermark"], "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, monochrome, black and white, low contrast, realistic, photorealistic, plain, simple, text, watermark", "ugly, deformed, noisy, blurry"], "color": "#322", "bgcolor": "#533"},
    {"id": 5, "type": "EmptyLatentImage", "pos": [340, 800], "size": {"0": 210, "1": 110}, "flags": {}, "order": 12, "mode": 0, "outputs": [{"name": "LATENT", "type": "LATENT", "links": [431], "slot_index": 0, "label": "Latent"}], "properties": {"Node name for S&R": "EmptyLatentImage"}, "widgets_values": [1024, 1024, 1]}
  ],
  "links": [
    [6, 7, 0, 3, 2, "CONDITIONING"], [7, 3, 0, 8, 0, "LATENT"], [48, 6, 0, 3, 1, "CONDITIONING"], [98, 65, 0, 6, 1, "STRING"], [100, 65, 1, 7, 1, "STRING"], [134, 65, 0, 92, 0, "STRING"], [135, 65, 1, 93, 0, "STRING"], [204, 126, 0, 65, 0, "STRING"], [205, 127, 0, 65, 1, "STRING"], [208, 129, 0, 132, 0, "LATENT"], [223, 91, 0, 3, 0, "MODEL"], [224, 91, 1, 6, 0, "CLIP"], [225, 91, 1, 7, 0, "CLIP"], [282, 172, 0, 179, 3, "INT"], [289, 172, 0, 181, 3, "INT"], [290, 181, 0, 176, 0, "IMAGE"], [361, 179, 0, 211, 0, "IMAGE"], [405, 181, 0, 179, 0, "IMAGE"], [416, 220, 0, 221, 0, "IMAGE"], [422, 222, 0, 223, 0, "IMAGE"], [431, 5, 0, 3, 3, "LATENT"], [434, 227, 0, 181, 1, "IMAGE"], [435, 227, 0, 179, 1, "IMAGE"], [436, 227, 0, 220, 1, "IMAGE"], [437, 227, 0, 222, 1, "IMAGE"], [454, 91, 0, 249, 0, "MODEL"], [457, 91, 1, 245, 0, "CLIP"], [458, 91, 1, 246, 0, "CLIP"], [459, 245, 0, 249, 1, "CONDITIONING"], [460, 246, 0, 249, 2, "CONDITIONING"], [461, 249, 0, 248, 0, "LATENT"], [464, 65, 0, 245, 1, "STRING"], [465, 65, 1, 246, 1, "STRING"], [466, 132, 0, 249, 3, "LATENT"], [487, 248, 0, 263, 0, "IMAGE"], [489, 8, 0, 264, 0, "IMAGE"], [491, 206, 0, 132, 1, "MASK"], [493, 206, 0, 266, 0, "MASK"], [497, 268, 0, 249, 4, "FLOAT"], [498, 270, 0, 249, 5, "COMBO"], [499, 271, 0, 249, 6, "COMBO"], [500, 272, 0, 249, 7, "INT"], [501, 274, 0, 249, 8, "INT"], [502, 268, 0, 3, 6, "FLOAT"], [503, 274, 0, 3, 5, "INT"], [504, 270, 0, 3, 7, "COMBO"], [505, 271, 0, 3, 8, "COMBO"], [506, 272, 0, 3, 4, "INT"], [609, 91, 2, 306, 0, "*"], [613, 306, 0, 307, 0, "*"], [614, 307, 0, 129, 1, "VAE"], [615, 307, 0, 248, 1, "VAE"], [617, 306, 0, 8, 1, "VAE"], [633, 317, 0, 312, 0, "IMAGE"], [634, 317, 0, 312, 1, "IMAGE"], [635, 316, 0, 312, 3, "INT"], [636, 312, 0, 313, 0, "IMAGE"], [637, 317, 0, 313, 1, "IMAGE"], [638, 316, 0, 313, 3, "INT"], [639, 318, 0, 314, 0, "IMAGE"], [642, 313, 0, 318, 0, "IMAGE"], [643, 317, 0, 318, 1, "IMAGE"], [644, 316, 0, 318, 3, "INT"], [645, 316, 0, 318, 4, "INT"], [659, 286, 0, 290, 0, "IMAGE"], [660, 288, 0, 291, 0, "IMAGE"], [669, 179, 0, 222, 0, "IMAGE"], [670, 222, 0, 220, 0, "IMAGE"], [671, 220, 0, 129, 0, "IMAGE"], [675, 322, 0, 323, 0, "IMAGE"], [676, 266, 0, 322, 0, "IMAGE"], [677, 323, 0, 324, 0, "IMAGE"], [683, 324, 0, 312, 2, "MASK"], [684, 324, 0, 313, 2, "MASK"], [685, 324, 0, 318, 2, "MASK"], [687, 248, 0, 317, 0, "*"], [688, 314, 0, 327, 0, "IMAGE"], [689, 327, 0, 315, 0, "IMAGE"], [690, 327, 0, 286, 0, "IMAGE"], [691, 327, 0, 288, 0, "IMAGE"], [692, 327, 0, 319, 0, "*"], [693, 328, 0, 181, 0, "IMAGE"], [695, 8, 0, 227, 0, "*"]
  ],
  "groups": [
    {"title": "\u7b2c\u4e00\u6b65\uff1a\u51fa\u56fe", "bounding": [77, 93, 1313, 862], "color": "#3f789e", "font_size": 24, "locked": false},
    {"title": "\u7b2c\u4e8c\u6b65\uff1a\u62fc\u56fe", "bounding": [1428, 94, 1269, 862], "color": "#3f789e", "font_size": 24, "locked": false},
    {"title": "\u7b2c\u4e09\u6b65\uff1a\u6d88\u9664\u63a5\u7f1d", "bounding": [78, 992, 958, 771], "color": "#3f789e", "font_size": 24, "locked": false},
    {"title": "\u7b2c\u56db\u6b65\uff1a\u8f93\u51fa\u989c\u8272\u8d34\u56fe", "bounding": [1076, 993, 1266, 775], "color": "#3f789e", "font_size": 24, "locked": false},
    {"title": "\u7b2c\u4e94\u6b65\uff1a\u8f93\u51fa\u6cd5\u7ebf\u548c\u6df1\u5ea6\u8d34\u56fe", "bounding": [2371, 990, 932, 773], "color": "#3f789e", "font_size": 24, "locked": false}
  ],
  "config": {}, "extra": {}, "version": 0.4}
```

```json [无缝贴图-AI生成v3.1]
{"last_node_id": 365, "last_link_id": 769, "nodes": [{"id": 245, "type": "CLIPTextEncode", "pos": [2670, 210], "size": {"0": 210, "1": 90}, "flags": {"collapsed": true}, "order": 14, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 457, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 464, "widget": {"name": "text"}, "label": "\u6587\u672c", "slot_index": 1}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [459], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["1girl"], "color": "#232", "bgcolor": "#353"}, {"id": 290, "type": "SaveImage", "pos": [1622.586564967992, 1231.5679835842157], "size": {"0": 390, "1": 460}, "flags": {}, "order": 45, "mode": 2, "inputs": [{"name": "images", "type": "IMAGE", "link": 650, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["normal-map"]}, {"id": 291, "type": "SaveImage", "pos": [2072.5865649680004, 1221.5679835842157], "size": {"0": 380, "1": 460}, "flags": {}, "order": 46, "mode": 2, "inputs": [{"name": "images", "type": "IMAGE", "link": 659, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["bump-map"]}, {"id": 288, "type": "AIO_Preprocessor", "pos": [2082.586564968002, 1081.567983584216], "size": {"0": 310, "1": 82}, "flags": {}, "order": 43, "mode": 2, "inputs": [{"name": "image", "type": "IMAGE", "link": 761, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [659], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "AIO_Preprocessor"}, "widgets_values": ["MiDaS-DepthMapPreprocessor", 1024]}, {"id": 315, "type": "ImageScale", "pos": [594.8780563964839, 168.33799871826173], "size": {"0": 210, "1": 130}, "flags": {}, "order": 11, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 665, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [693], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageScale"}, "widgets_values": ["bilinear", 1024, 1024, "center"]}, {"id": 286, "type": "AIO_Preprocessor", "pos": [1632.586564967992, 1091.567983584216], "size": {"0": 310, "1": 82}, "flags": {}, "order": 42, "mode": 2, "inputs": [{"name": "image", "type": "IMAGE", "link": 760, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [650], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "AIO_Preprocessor"}, "widgets_values": ["BAE-NormalMapPreprocessor", 1024]}, {"id": 337, "type": "Reroute", "pos": [840, 170], "size": [75, 26], "flags": {}, "order": 17, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 693, "label": ""}], "outputs": [{"name": "", "type": "IMAGE", "links": [674, 677, 681, 684, 767], "slot_index": 0}], "properties": {"showOutputText": false, "horizontal": false}}, {"id": 330, "type": "Image Overlay", "pos": [590, 350], "size": {"0": 210, "1": 290}, "flags": {}, "order": 19, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 763, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 677, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "y_offset", "type": "INT", "link": 678, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb", "slot_index": 4}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [672, 673], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 0, 1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 328, "type": "PreviewImage", "pos": [590, 680], "size": {"0": 210, "1": 250}, "flags": {}, "order": 22, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 672, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 332, "type": "PreviewImage", "pos": [830, 680], "size": {"0": 210, "1": 250}, "flags": {}, "order": 25, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 679, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 329, "type": "Image Overlay", "pos": [830, 300], "size": {"0": 210, "1": 290}, "flags": {}, "order": 23, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 673, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 674, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 675, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb", "slot_index": 3}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [679, 683], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 1020, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 335, "type": "Image Overlay", "pos": [1060, 300], "size": {"0": 210, "1": 290}, "flags": {}, "order": 26, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 683, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 684, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [680, 685], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 1020, 1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 333, "type": "Image Overlay", "pos": [1300, 300], "size": {"0": 210, "1": 290}, "flags": {}, "order": 27, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 680, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 681, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": null, "label": "\u906e\u7f69", "slot_index": 2}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [682, 689], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 0, 0, 0, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 336, "type": "PreviewImage", "pos": [1070, 680], "size": {"0": 210, "1": 250}, "flags": {}, "order": 28, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 685, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 334, "type": "PreviewImage", "pos": [1310, 680], "size": {"0": 210, "1": 250}, "flags": {}, "order": 29, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 682, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 255, "type": "Image Overlay", "pos": [590, 1430], "size": {"0": 210, "1": 290}, "flags": {}, "order": 37, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 471, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 619, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 755, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "y_offset", "type": "INT", "link": 495, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [668], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 1280, 256, 0, -1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 265, "type": "SaveImage", "pos": [1070, 1260], "size": {"0": 440, "1": 460}, "flags": {}, "order": 44, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 762, "label": "\u56fe\u50cf"}], "properties": {}, "widgets_values": ["color-map"]}, {"id": 251, "type": "Image Overlay", "pos": [600, 1070], "size": {"0": 210, "1": 290}, "flags": {}, "order": 36, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 753, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 618, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 756, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 496, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [471], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 256, 1280, -1020, 0, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 280, "type": "Reroute", "pos": [470, 1030], "size": [75, 26], "flags": {}, "order": 35, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 754, "label": ""}], "outputs": [{"name": "", "type": "IMAGE", "links": [560, 618, 619, 753], "slot_index": 0, "label": ""}], "properties": {"showOutputText": false, "horizontal": false}}, {"id": 282, "type": "Image Overlay", "pos": [840, 1300], "size": {"0": 210, "1": 290}, "flags": {}, "order": 38, "mode": 0, "inputs": [{"name": "base_image", "type": "IMAGE", "link": 668, "label": "\u57fa\u7840\u56fe\u50cf", "slot_index": 0}, {"name": "overlay_image", "type": "IMAGE", "link": 560, "label": "\u8986\u76d6\u56fe\u50cf", "slot_index": 1}, {"name": "optional_mask", "type": "MASK", "link": 757, "label": "\u906e\u7f69", "slot_index": 2}, {"name": "x_offset", "type": "INT", "link": 550, "widget": {"name": "x_offset"}, "label": "X\u504f\u79fb", "slot_index": 3}, {"name": "y_offset", "type": "INT", "link": 551, "widget": {"name": "y_offset"}, "label": "Y\u504f\u79fb", "slot_index": 4}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [563], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "Image Overlay"}, "widgets_values": ["None", "nearest-exact", 1, 256, 256, -1020, -1020, 0, 0], "color": "#222233", "bgcolor": "#333355", "shape": 1}, {"id": 260, "type": "ImageCrop", "pos": [840, 1080], "size": {"0": 210, "1": 130}, "flags": {}, "order": 39, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 563, "label": "\u56fe\u50cf", "slot_index": 0}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [758], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageCrop"}, "widgets_values": [1020, 1020, 0, 0]}, {"id": 308, "type": "PlaySound|pysssss", "pos": [1310, 1090], "size": {"0": 210, "1": 110}, "flags": {}, "order": 41, "mode": 0, "inputs": [{"name": "any", "type": "*", "link": 759, "label": "\u8f93\u5165"}], "outputs": [{"name": "*", "type": "*", "links": null, "shape": 6}], "properties": {"Node name for S&R": "PlaySound|pysssss"}, "widgets_values": ["always", 1, "notify.mp3"]}, {"id": 360, "type": "ImageToMask", "pos": [340, 1690], "size": {"0": 210, "1": 60}, "flags": {}, "order": 24, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 750, "label": "\u56fe\u50cf"}], "outputs": [{"name": "MASK", "type": "MASK", "links": [755, 756, 757], "shape": 3, "label": "\u906e\u7f69", "slot_index": 0}], "properties": {"Node name for S&R": "ImageToMask"}, "widgets_values": ["red"]}, {"id": 359, "type": "ImageCrop", "pos": [340, 1510], "size": {"0": 210, "1": 130}, "flags": {}, "order": 21, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 748, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [750], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageCrop"}, "widgets_values": [1276, 1276, 0, 0]}, {"id": 357, "type": "ImageScale", "pos": [340, 1330], "size": {"0": 210, "1": 130}, "flags": {}, "order": 18, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 746, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [748], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageScale"}, "widgets_values": ["bilinear", 1590, 1590, "disabled"]}, {"id": 267, "type": "PrimitiveNode", "pos": [340, 1100], "size": {"0": 210, "1": 80}, "flags": {}, "order": 0, "mode": 0, "outputs": [{"name": "INT", "type": "INT", "links": [495, 496, 550, 551], "slot_index": 0, "widget": {"name": "y_offset"}, "label": "INT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [-1020, "fixed"]}, {"id": 266, "type": "MaskToImage", "pos": [400, 1240], "size": {"0": 140, "1": 30}, "flags": {}, "order": 12, "mode": 0, "inputs": [{"name": "mask", "type": "MASK", "link": 493, "label": "\u906e\u7f69"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [746], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "MaskToImage"}}, {"id": 361, "type": "ImageScale", "pos": [1080, 1080], "size": {"0": 210, "1": 130}, "flags": {}, "order": 40, "mode": 0, "inputs": [{"name": "image", "type": "IMAGE", "link": 758, "label": "\u56fe\u50cf"}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [759, 760, 761, 762], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "ImageScale"}, "widgets_values": ["bilinear", 1024, 1024, "disabled"]}, {"id": 126, "type": "DeepTranslatorTextNode", "pos": [1890, 190], "size": [220, 260], "flags": {}, "order": 1, "mode": 0, "outputs": [{"name": "text", "type": "STRING", "links": [204], "shape": 3, "label": "\u6587\u672c", "slot_index": 0}], "properties": {"Node name for S&R": "DeepTranslatorTextNode"}, "widgets_values": ["auto", "english", "disable", "", "", "GoogleTranslator [free]", "rock", "proxy_hide", "authorization_hide"], "color": "#232", "bgcolor": "#353"}, {"id": 249, "type": "KSampler", "pos": [2370, 430], "size": {"0": 260, "1": 470}, "flags": {}, "order": 32, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 768, "label": "\u6a21\u578b"}, {"name": "positive", "type": "CONDITIONING", "link": 459, "label": "\u6b63\u9762\u6761\u4ef6", "slot_index": 1}, {"name": "negative", "type": "CONDITIONING", "link": 460, "label": "\u8d1f\u9762\u6761\u4ef6", "slot_index": 2}, {"name": "latent_image", "type": "LATENT", "link": 466, "label": "Latent", "slot_index": 3}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [461], "slot_index": 0, "label": "Latent"}], "properties": {"Node name for S&R": "KSampler"}, "widgets_values": [1086404413043726, "randomize", 8, 1, "euler", "sgm_uniform", 1]}, {"id": 246, "type": "CLIPTextEncode", "pos": [2680, 250], "size": {"0": 210, "1": 100}, "flags": {"collapsed": true}, "order": 16, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 458, "label": "CLIP"}, {"name": "text", "type": "STRING", "link": 465, "widget": {"name": "text"}, "label": "\u6587\u672c", "slot_index": 1}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [460], "slot_index": 0, "label": "\u6761\u4ef6"}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": ["text, watermark"], "color": "#322", "bgcolor": "#533"}, {"id": 65, "type": "SDXLPromptStyler", "pos": [2380, 180], "size": {"0": 230, "1": 170}, "flags": {"collapsed": false}, "order": 8, "mode": 0, "inputs": [{"name": "text_positive", "type": "STRING", "link": 204, "widget": {"name": "text_positive"}, "label": "\u6b63\u9762\u6761\u4ef6"}, {"name": "text_negative", "type": "STRING", "link": 205, "widget": {"name": "text_negative"}, "label": "\u8d1f\u9762\u6761\u4ef6"}], "outputs": [{"name": "positive_prompt_text_g", "type": "STRING", "links": [134, 464], "shape": 3, "label": "positive_prompt_text_g", "slot_index": 0}, {"name": "negative_prompt_text_g", "type": "STRING", "links": [135, 465], "shape": 3, "label": "negative_prompt_text_g", "slot_index": 1}], "properties": {"Node name for S&R": "SDXLPromptStyler"}, "widgets_values": ["1 girl, long hair, dress, 3/4 profile, ", "text, watermark", "sai-texture", true, true, true], "color": "#232", "bgcolor": "#353"}, {"id": 127, "type": "DeepTranslatorTextNode", "pos": [2130, 190], "size": [210, 260], "flags": {}, "order": 2, "mode": 0, "outputs": [{"name": "text", "type": "STRING", "links": [205], "shape": 3, "label": "\u6587\u672c", "slot_index": 0}], "properties": {"Node name for S&R": "DeepTranslatorTextNode"}, "widgets_values": ["auto", "english", "disable", "", "", "GoogleTranslator [free]", "", "proxy_hide", "authorization_hide"], "color": "#322", "bgcolor": "#533"}, {"id": 248, "type": "VAEDecode", "pos": [2680, 330], "size": {"0": 140, "1": 50}, "flags": {}, "order": 33, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 461, "label": "Latent"}, {"name": "vae", "type": "VAE", "link": 615, "label": "VAE", "slot_index": 1}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [487, 754], "slot_index": 0, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "VAEDecode"}}, {"id": 263, "type": "PreviewImage", "pos": [2660, 450], "size": {"0": 310, "1": 350}, "flags": {}, "order": 34, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 487, "label": "\u56fe\u50cf"}], "properties": {"Node name for S&R": "PreviewImage"}}, {"id": 363, "type": "EmptyImage", "pos": [350, 540], "size": {"0": 210, "1": 130}, "flags": {}, "order": 3, "mode": 0, "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [763], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}], "properties": {"Node name for S&R": "EmptyImage"}, "widgets_values": [1276, 1276, 1, 999999]}, {"id": 327, "type": "PrimitiveNode", "pos": [350, 750], "size": {"0": 210, "1": 80}, "flags": {}, "order": 4, "mode": 0, "outputs": [{"name": "INT", "type": "INT", "links": [675, 678], "slot_index": 0, "widget": {"name": "x_offset"}, "label": "INT"}], "properties": {"Run widget replace on values": false}, "widgets_values": [1020, "fixed"]}, {"id": 91, "type": "CheckpointLoader|pysssss", "pos": [1590, 200], "size": {"0": 210, "1": 122}, "flags": {}, "order": 5, "mode": 0, "outputs": [{"name": "MODEL", "type": "MODEL", "links": [766], "shape": 3, "label": "\u6a21\u578b", "slot_index": 0}, {"name": "CLIP", "type": "CLIP", "links": [457, 458], "shape": 3, "label": "CLIP", "slot_index": 1}, {"name": "VAE", "type": "VAE", "links": [634], "shape": 3, "label": "VAE", "slot_index": 2}], "properties": {"Node name for S&R": "CheckpointLoader|pysssss"}, "widgets_values": [{"content": "SDXL-lightning/sdxl_lightning_8step.safetensors", "image": null}, "[none]"]}, {"id": 326, "type": "LoadImage", "pos": [344.878056396484, 178.33799871826173], "size": [220, 310], "flags": {}, "order": 6, "mode": 0, "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [665], "shape": 3, "label": "\u56fe\u50cf", "slot_index": 0}, {"name": "MASK", "type": "MASK", "links": null, "shape": 3, "label": "\u906e\u7f69"}], "properties": {"Node name for S&R": "LoadImage"}, "widgets_values": ["u=3985828024,2950789954&fm=193.jpeg", "image"]}, {"id": 206, "type": "LoadImageMask", "pos": [1600, 600], "size": {"0": 220, "1": 320}, "flags": {}, "order": 7, "mode": 0, "outputs": [{"name": "MASK", "type": "MASK", "links": [491, 493], "shape": 3, "label": "\u906e\u7f69", "slot_index": 0}], "properties": {"Node name for S&R": "LoadImageMask"}, "widgets_values": ["alpha \u8d34\u56fe.png", "alpha", "image"]}, {"id": 129, "type": "VAEEncode", "pos": [1650, 450], "size": {"0": 140, "1": 46}, "flags": {}, "order": 30, "mode": 0, "inputs": [{"name": "pixels", "type": "IMAGE", "link": 689, "label": "\u56fe\u50cf"}, {"name": "vae", "type": "VAE", "link": 614, "label": "VAE", "slot_index": 1}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [208], "shape": 3, "label": "Latent", "slot_index": 0}], "properties": {"Node name for S&R": "VAEEncode"}}, {"id": 92, "type": "ShowText|pysssss", "pos": [1890, 500], "size": {"0": 210, "1": 130}, "flags": {"collapsed": false}, "order": 13, "mode": 0, "inputs": [{"name": "text", "type": "STRING", "link": 134, "widget": {"name": "text"}, "label": "\u6587\u672c"}], "outputs": [{"name": "STRING", "type": "STRING", "links": null, "shape": 6, "label": "\u5b57\u7b26\u4e32"}], "properties": {"Node name for S&R": "ShowText|pysssss"}, "widgets_values": [["ethereal fantasy concept art of  psychedelic style 1 girl, \u9648\u5c0f\u7ead, long hair, dress, looking back,   . vibrant colors, swirling patterns, abstract forms, surreal, trippy . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy"], "ethereal fantasy concept art of  psychedelic style 1 girl, \u9648\u5c0f\u7ead, long hair, dress, looking back,   . vibrant colors, swirling patterns, abstract forms, surreal, trippy . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy", "texture rock top down close-up"], "color": "#232", "bgcolor": "#353"}, {"id": 93, "type": "ShowText|pysssss", "pos": [2120, 510], "size": {"0": 220, "1": 130}, "flags": {"collapsed": false}, "order": 15, "mode": 0, "inputs": [{"name": "text", "type": "STRING", "link": 135, "widget": {"name": "text"}, "label": "\u6587\u672c"}], "outputs": [{"name": "STRING", "type": "STRING", "links": null, "shape": 6, "label": "\u5b57\u7b26\u4e32"}], "properties": {"Node name for S&R": "ShowText|pysssss"}, "widgets_values": [["photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, monochrome, black and white, low contrast, realistic, photorealistic, plain, simple, text, watermark"], "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white, monochrome, black and white, low contrast, realistic, photorealistic, plain, simple, text, watermark", "ugly, deformed, noisy, blurry"], "color": "#322", "bgcolor": "#533"}, {"id": 365, "type": "IPAdapterUnifiedLoader", "pos": [1870, 730], "size": {"0": 240, "1": 80}, "flags": {"collapsed": false}, "order": 9, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 766, "label": "model", "slot_index": 0}, {"name": "ipadapter", "type": "IPADAPTER", "link": null, "label": "ipadapter"}], "outputs": [{"name": "model", "type": "MODEL", "links": [769], "shape": 3, "label": "model"}, {"name": "ipadapter", "type": "IPADAPTER", "links": [764], "shape": 3, "label": "ipadapter"}], "properties": {"Node name for S&R": "IPAdapterUnifiedLoader"}, "widgets_values": ["PLUS (high strength)"]}, {"id": 364, "type": "IPAdapter", "pos": [2130, 730], "size": {"0": 210, "1": 170}, "flags": {}, "order": 20, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 769, "label": "\u6a21\u578b", "slot_index": 0}, {"name": "ipadapter", "type": "IPADAPTER", "link": 764, "slot_index": 1}, {"name": "image", "type": "IMAGE", "link": 767, "label": "\u56fe\u50cf", "slot_index": 2}, {"name": "attn_mask", "type": "MASK", "link": null}], "outputs": [{"name": "MODEL", "type": "MODEL", "links": [768], "shape": 3, "label": "\u6a21\u578b", "slot_index": 0}], "properties": {"Node name for S&R": "IPAdapter"}, "widgets_values": [0.8, 0, 1]}, {"id": 132, "type": "SetLatentNoiseMask", "pos": [1880, 870], "size": {"0": 140, "1": 50}, "flags": {}, "order": 31, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 208, "label": "Latent"}, {"name": "mask", "type": "MASK", "link": 491, "label": "\u906e\u7f69", "slot_index": 1}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [466], "shape": 3, "label": "Latent", "slot_index": 0}], "properties": {"Node name for S&R": "SetLatentNoiseMask"}}, {"id": 307, "type": "Reroute", "pos": [1710, 360], "size": [75, 26], "flags": {}, "order": 10, "mode": 0, "inputs": [{"name": "", "type": "*", "link": 634}], "outputs": [{"name": "", "type": "VAE", "links": [614, 615], "slot_index": 0}], "properties": {"showOutputText": false, "horizontal": false}}], "links": [[134, 65, 0, 92, 0, "STRING"], [135, 65, 1, 93, 0, "STRING"], [204, 126, 0, 65, 0, "STRING"], [205, 127, 0, 65, 1, "STRING"], [208, 129, 0, 132, 0, "LATENT"], [457, 91, 1, 245, 0, "CLIP"], [458, 91, 1, 246, 0, "CLIP"], [459, 245, 0, 249, 1, "CONDITIONING"], [460, 246, 0, 249, 2, "CONDITIONING"], [461, 249, 0, 248, 0, "LATENT"], [464, 65, 0, 245, 1, "STRING"], [465, 65, 1, 246, 1, "STRING"], [466, 132, 0, 249, 3, "LATENT"], [471, 251, 0, 255, 0, "IMAGE"], [487, 248, 0, 263, 0, "IMAGE"], [491, 206, 0, 132, 1, "MASK"], [493, 206, 0, 266, 0, "MASK"], [495, 267, 0, 255, 3, "INT"], [496, 267, 0, 251, 3, "INT"], [550, 267, 0, 282, 3, "INT"], [551, 267, 0, 282, 4, "INT"], [560, 280, 0, 282, 1, "IMAGE"], [563, 282, 0, 260, 0, "IMAGE"], [614, 307, 0, 129, 1, "VAE"], [615, 307, 0, 248, 1, "VAE"], [618, 280, 0, 251, 1, "IMAGE"], [619, 280, 0, 255, 1, "IMAGE"], [634, 91, 2, 307, 0, "*"], [650, 286, 0, 290, 0, "IMAGE"], [659, 288, 0, 291, 0, "IMAGE"], [665, 326, 0, 315, 0, "IMAGE"], [668, 255, 0, 282, 0, "IMAGE"], [672, 330, 0, 328, 0, "IMAGE"], [673, 330, 0, 329, 0, "IMAGE"], [674, 337, 0, 329, 1, "IMAGE"], [675, 327, 0, 329, 3, "INT"], [677, 337, 0, 330, 1, "IMAGE"], [678, 327, 0, 330, 3, "INT"], [679, 329, 0, 332, 0, "IMAGE"], [680, 335, 0, 333, 0, "IMAGE"], [681, 337, 0, 333, 1, "IMAGE"], [682, 333, 0, 334, 0, "IMAGE"], [683, 329, 0, 335, 0, "IMAGE"], [684, 337, 0, 335, 1, "IMAGE"], [685, 335, 0, 336, 0, "IMAGE"], [689, 333, 0, 129, 0, "IMAGE"], [693, 315, 0, 337, 0, "*"], [746, 266, 0, 357, 0, "IMAGE"], [748, 357, 0, 359, 0, "IMAGE"], [750, 359, 0, 360, 0, "IMAGE"], [753, 280, 0, 251, 0, "IMAGE"], [754, 248, 0, 280, 0, "*"], [755, 360, 0, 255, 2, "MASK"], [756, 360, 0, 251, 2, "MASK"], [757, 360, 0, 282, 2, "MASK"], [758, 260, 0, 361, 0, "IMAGE"], [759, 361, 0, 308, 0, "*"], [760, 361, 0, 286, 0, "IMAGE"], [761, 361, 0, 288, 0, "IMAGE"], [762, 361, 0, 265, 0, "IMAGE"], [763, 363, 0, 330, 0, "IMAGE"], [764, 365, 1, 364, 1, "IPADAPTER"], [766, 91, 0, 365, 0, "MODEL"], [767, 337, 0, 364, 2, "IMAGE"], [768, 364, 0, 249, 0, "MODEL"], [769, 365, 0, 364, 0, "MODEL"]], "groups": [{"title": "\u7b2c\u4e00\u6b65\uff1a\u4e0a\u4f20\u7d20\u6750+\u62fc\u56fe", "bounding": [305, 85, 1242, 869], "color": "#3f789e", "font_size": 24, "locked": false}, {"title": "\u7b2c\u4e8c\u6b65\uff1a\u6d88\u9664\u63a5\u7f1d", "bounding": [1574, 86, 1419, 865], "color": "#3f789e", "font_size": 24, "locked": false}, {"title": "\u7b2c\u4e09\u6b65\uff1a\u8f93\u51fa\u989c\u8272\u8d34\u56fe", "bounding": [301, 977, 1246, 793], "color": "#3f789e", "font_size": 24, "locked": false}, {"title": "\u7b2c\u56db\u6b65\uff1a\u8f93\u51fa\u6df1\u5ea6\u56fe+\u6cd5\u7ebf\u56fe", "bounding": [1576, 981, 907, 791], "color": "#3f789e", "font_size": 24, "locked": false}], "config": {}, "extra": {}, "version": 0.4}
```
:::

涉及工具/节点：
- ipadapter地址：https://github.com/cubiq/ComfyUI_IPAdapter_plus
- 效率节点地址：https://github.com/jags111/efficiency-nodes-comfyui
- SDXL风格选择插件（汉化版）https://github.com/ZHO-ZHO-ZHO/sdxl_prompt_styler-Zh-Chinese
- https://github.com/bash-j/mikey_nodes

Model:
- https://huggingface.co/ByteDance/SDXL-Lightning

结合ComfyUI_IPAdapter_plus项目的`IPAdapterPlus.py`和`utils.py`(2024.5)
- 将`CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`放入`models\clip_vision`中
- 将`ip-adapter-plus_sdxl_vit-h.safetensors`放入`models\ipadapter`中(if folder not exist, create one)


## 混元-DiT

[Project](https://dit.hunyuan.tencent.com/) | [Paper](https://arxiv.org/abs/2405.08748) | [Model](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) | [Code](https://github.com/tencent/HunyuanDiT)

混元-DiT是腾讯提出的一个支持中英文生成图片的模型。

### CLIP model
使用了一种bilingual CLIP。

在`hydit/inference.py`中，有：
```python
#...
from transformers import BertModel, BertTokenizer
# ...
from .diffusion.pipeline import StableDiffusionPipeline
# ...

def get_pipeline(args, vae, text_encoder, tokenizer, model, device, rank,
                 embedder_t5, infer_mode, sampler=None):
    #...
    pipeline = StableDiffusionPipeline(vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       unet=model,
                                       scheduler=scheduler,
                                       feature_extractor=None,
                                       safety_checker=None,
                                       requires_safety_checker=False,
                                       progress_bar_config=progress_bar_config,
                                       embedder_t5=embedder_t5,
                                       infer_mode=infer_mode,
                                       )

    pipeline = pipeline.to(device)

    return pipeline, sampler
# ...
class End2End(object):
    def __init__(self, args, models_root_path):
        # ...
        # ========================================================================
        logger.info(f"Loading CLIP Text Encoder...")
        text_encoder_path = self.root / "clip_text_encoder"
        self.clip_text_encoder = BertModel.from_pretrained(str(text_encoder_path), False, revision=None).to(self.device)
        logger.info(f"Loading CLIP Text Encoder finished")

        # ========================================================================
        logger.info(f"Loading CLIP Tokenizer...")
        tokenizer_path = self.root / "tokenizer"
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        logger.info(f"Loading CLIP Tokenizer finished")
        # ...
        self.pipeline, self.sampler = self.load_sampler()
        # ...

    def load_sampler(self, sampler=None):
        pipeline, sampler = get_pipeline(self.args,
                                         self.vae,
                                         self.clip_text_encoder,
                                         self.tokenizer,
                                         self.model,
                                         device=self.device,
                                         rank=0,
                                         embedder_t5=self.embedder_t5,
                                         infer_mode=self.infer_mode,
                                         sampler=sampler,
                                         )
        return pipeline, sampler
    # ...
    def predict(self,
                user_prompt,
                height=1024,
                width=1024,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=100,
                guidance_scale=6,
                batch_size=1,
                src_size_cond=(1024, 1024),
                sampler=None,
                ):
        # ...

        samples = self.pipeline(
            height=target_height,
            width=target_width,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            image_meta_size=image_meta_size,
            style=style,
            return_dict=False,
            generator=generator,
            freqs_cis_img=freqs_cis_img,
            use_fp16=self.args.use_fp16,
            learn_sigma=self.args.learn_sigma,
        )[0]
        gen_time = time.time() - start_time
        logger.debug(f"Success, time: {gen_time}")

        return {
            'images': samples,
            'seed': seed,
        }
```
在`hydit/diffusion/pipeline.py`中，有：
```python
# ...
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
# ...
class StableDiffusionPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin):
    # ...
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: Union[BertModel, CLIPTextModel],
        tokenizer: Union[BertTokenizer, CLIPTokenizer],
        unet: Union[HunYuanDiT, UNet2DConditionModel],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
        progress_bar_config: Dict[str, Any] = None,
        embedder_t5=None,
        infer_mode='torch',
    ):
        super().__init__()
        # ...
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # ...
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            height: int,
            width: int,
            prompt: Union[str, List[str]] = None,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            prompt_embeds_t5: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds_t5: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            image_meta_size: Optional[torch.LongTensor] = None,
            style: Optional[torch.LongTensor] = None,
            progress: bool = True,
            use_fp16: bool = False,
            freqs_cis_img: Optional[tuple] = None,
            learn_sigma: bool = True,
    ):
        # ...
```

再后面查下去就到diffuser内部库了（用了很多继承的变量和方法），综上一顿瞎分析可知，如果想单独使用“CLIP”模型，先下载[Model](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) 中`clip_text_encoder`和`tokenizer`的模型：
```python
import torch
from transformers import BertTokenizer, BertModel

# 设置模型和Tokenizer的路径
model_dir = "hunyuanDiT/clip_text_encoder"  # 包含config.json和pytorch_model.bin的目录
tokenizer_dir = "hunyuanDiT/tokenizer"  # 包含tokenizer文件的目录

# 加载Tokenizer和模型
tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
model = BertModel.from_pretrained(model_dir)

# 准备输入文本
text = "你好，世界！"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 输出结果
last_hidden_state = outputs.last_hidden_state
print(last_hidden_state)
print("Shape:", last_hidden_state.shape)


```

在`dialoggen/llava/model/multimodal_encoder/clip_encoder.py`中
```python
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

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

DreamScene360: [Paper](https://arxiv.org/abs/2404.06903)

Text2Room: [Project](https://lukashoel.github.io/text-to-room/) | [Code](https://github.com/lukasHoel/text2room)

Text2NeRF: [Project](https://eckertzhang.github.io/Text2NeRF.github.io/) | [Code](https://github.com/eckertzhang/Text2NeRF)

GaussianCube: [Project](https://gaussiancube.github.io/) | [Paper](https://arxiv.org/abs/2403.19655) | [Code](https://github.com/GaussianCube/GaussianCube)

### CAT3D
[Project](https://cat3d.github.io/) | [Paper](https://arxiv.org/abs/2405.10314)

之前的工作是侧重于如何更好地重建模型/提升单图重建模型的质量，但这篇文章的侧重点是如何通过diffusion model 产生更多视角的图像，解决最大的痛点。

<div class="theme-image">
  <img src="./assets/CAT3D.png" alt="Light Mode Image" class="light-mode">
  <img src="./assets/dark_CAT3D.png" alt="Dark Mode Image" class="dark-mode">
</div>

CAT3D has two stages:

(1) generate a large set of synthetic views from a **multi-view latent diffusion model** conditioned on the input views alongside
the camera poses of target views;

(2) run a **robust 3D reconstruction pipeline** on the observed and
generated views to learn a NeRF representation.

CAT3D最终可以通过多张图像、单张图像或纯文本生成3D模型。

### LGM
[Project](https://me.kiui.moe/lgm/) | [Paper](https://arxiv.org/abs/2402.05054) | [Demo](https://huggingface.co/spaces/ashawkey/LGM)

推荐环境：CUDA 11.8以上



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
