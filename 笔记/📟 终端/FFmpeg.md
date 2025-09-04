---
tags:
  - 视频剪辑
---

# FFmpeg

FFmpeg 和 OpenCV 都是用于处理图像和视频的流行工具，但它们的主要功能和用途有所不同。

1. **FFmpeg**:
- 主要用于音视频处理，可以进行视频压缩、格式转换、裁剪、合并等操作。
- 非常高效，特别是在批量处理大量视频文件时。
- 提供了丰富的命令行选项，可以精确控制处理过程。
- 缺点是需要一定的学习成本，特别是对于不熟悉命令行操作的用户来说。

2. **OpenCV**:
- 主要用于计算机视觉任务，如图像处理、对象检测、特征提取等。
- 提供了丰富的图像处理和计算机视觉算法，以及易于使用的 Python 接口。
- 功能强大且灵活，可以在 Python 环境中直接调用，并与其他 Python 库集成。
- 缺点是对于大规模视频处理可能不够高效，并且在某些特定视频处理任务上可能不如专门的视频处理工具效果好。

总的来说，如果您主要进行音视频处理任务，如视频压缩、格式转换等，建议使用 FFmpeg。而如果您需要进行图像处理或计算机视觉任务，如图像处理、对象检测等，建议使用 OpenCV。在某些情况下，两者也可以结合使用，以充分发挥各自的优势。

## Install
验证`ffmpeg -version`
### Linux
对于x86的系统
```shell
sudo nala install ffmpeg
```

### Windows
将下载的压缩文件解压，将`bin`文件夹添加到环境变量中

## Uninstall
- 对于 Ubuntu/Debian：
```shell
sudo nala remove ffmpeg # sudo apt-get remove ffmpeg
```

- 对于 CentOS/RHEL：
```shell
sudo yum remove ffmpeg
```

- 对于 macOS（使用 Homebrew）：
```shell
brew uninstall ffmpeg
```

**手动卸载**： 如果你是手动编译并安装的 FFmpeg，你可以通过找到安装目录并将相关文件删除来卸载它。通常情况下，FFmpeg 的可执行文件、库文件和头文件会安装在 `/usr/local/bin`、`/usr/local/lib` 和 `/usr/local/include` 等目录下。你可以使用 `rm` 命令删除这些文件，但请务必小心以免意外删除其他文件。

卸载完成后，你可以验证 FFmpeg 是否已成功移除，方法是尝试运行 `ffmpeg` 命令并检查是否会出现“未找到命令”的错误。

## 画面

### 旋转
将视频旋转90度
```shell
ffmpeg -i input.mp4 -vf "transpose=1" -c:a copy output.mp4
```
- `-i input.mp4` 指定输入视频文件。
- `-vf "transpose=1"` 应用 transpose 滤镜，其中 1 表示逆时针旋转90度。
- `-c:a copy` 表示将音频流直接复制到输出文件中，以保持音频质量不变。
- `output.mp4` 是输出文件的名称。

### 裁剪
裁剪视频的下方100像素
```shell
ffmpeg -i input.mp4 -vf "crop=in_w:in_h-100:0:100" -c:a copy output.mp4
```
- `in_w` 表示输入视频的宽度
- `in_h-100` 表示裁剪后的高度比原始高度少100像素

裁剪视频的下方100像素
```shell
ffmpeg -i input.mp4 -vf "crop=in_w:in_h-100:0:0" -c:a copy output.mp4
```

裁剪视频的左方100像素
```shell
ffmpeg -i input.mp4 -vf "crop=in_w-100:in_h:100:0" -c:a copy output.mp4
```

### 缩放
```shell
ffmpeg -i input_4k_video.mp4 -vf scale=1920:1080 -c:a copy output_1080p_video.mp4
```
- `-i input_4k_video.mp4`：指定输入视频文件。
- `-vf scale=1920:1080`：指定视频过滤器，将视频缩放到 1920x1080 的分辨率（1080p）。
- `-c:a copy`：指定音频编解码器为复制，表示音频不需要重新编码，直接复制到输出文件。
- `output_1080p_video.mp4`：指定输出视频文件名。

### 视频合并

合并视频并循环所有视频1次
```shell
ffmpeg \
-stream_loop 1 -i iPhone1.mp4 -stream_loop 1 -i iPhone5.mp4 -stream_loop 1 -i 餐具1.mp4 -stream_loop 1 -i 餐具3.mp4 \
-stream_loop 1 -i laptop1.mp4 -stream_loop 1 -i laptop2.mp4 -stream_loop 1 -i 茶具2.mp4 -stream_loop 1 -i 茶具3.mp4 \
-stream_loop 1 -i speaker1.mp4 -stream_loop 1 -i speaker2.mp4 -stream_loop 1 -i 花束1.mp4 -stream_loop 1 -i 花束4.mp4 \
-filter_complex "\
[0:v]scale=512:512[v0]; \
[1:v]scale=512:512[v1]; \
[2:v]scale=512:512[v2]; \
[3:v]scale=512:512[v3]; \
[4:v]scale=512:512[v4]; \
[5:v]scale=512:512[v5]; \
[6:v]scale=512:512[v6]; \
[7:v]scale=512:512[v7]; \
[8:v]scale=512:512[v8]; \
[9:v]scale=512:512[v9]; \
[10:v]scale=512:512[v10]; \
[11:v]scale=512:512[v11]; \
[v0][v1][v2][v3]hstack=inputs=4[top]; \
[v4][v5][v6][v7]hstack=inputs=4[middle]; \
[v8][v9][v10][v11]hstack=inputs=4[bottom]; \
[top][middle][bottom]vstack=inputs=3[output]" \
-map "[output]" -c:v libx264 output2.mp4
```

长宽缩放一半合并两个视频，保留第二个视频音频
```
ffmpeg -i origin.mp4 -i CN.mp4 -filter_complex "[0:v]scale=iw/2:ih/2[v0];[1:v]scale=iw/2:ih/2[v1];[v0][v1]hstack=inputs=2[outv]" -map "[outv]" -map 1:a -c:v libx264 -c:a aac -shortest output_half_audio2.mp4
```

### 图片拼接成视频
假设文件夹结构形如`1.png`, `2.png` ...
```shell
ffmpeg -framerate 3 -i "path/to/images/%d.png" -c:v libx264 -crf 0 -preset veryslow ori.mp4
```

如果文件夹结构形如`01.jpg`, `02.jpg` ...
```shell
ffmpeg -framerate 3 -i "path/to/images/%02d.jpg" -c:v libx264 -crf 0 -preset veryslow ori.mp4
```

## 常见问题
手动安装torchvision时：‘AV_CODEC_CAP_INTRA_ONLY’ was not declared in this scope; did you mean ‘AV_CODEC_PROP_INTRA_ONLY’?

解决方法：卸载FFmpeg，然后安好了torchvision之后再安回去

