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

四周裁剪，附带辅助计算参数的html
::: code-group
```shell [video_crop.bat]
@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============== 配置 ===============
set "INPUT=ビデオ_2025.mov"
set TOP=60
set BOTTOM=40
set LEFT=30
set RIGHT=50
:: ===================================

if not exist "%INPUT%" (
    echo ❌ ファイルが存在しません: %INPUT%
    pause
    exit /b 1
)

:: === 安全获取宽高（使用临时文件）===
set "WIDTH="
set "HEIGHT="

ffprobe -v error -select_streams v:0 -show_entries stream=width  -of default=nw=1:nk=1 "%INPUT%" >"%temp%\probe_w.txt" 2>nul
ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "%INPUT%" >"%temp%\probe_h.txt" 2>nul

for /f "usebackq delims=" %%a in ("%temp%\probe_w.txt") do set "WIDTH=%%a"
for /f "usebackq delims=" %%a in ("%temp%\probe_h.txt") do set "HEIGHT=%%a"

:: 清理临时文件
del "%temp%\probe_w.txt" "%temp%\probe_h.txt" >nul 2>&1

:: 验证
if "%WIDTH%"=="" (
    echo ❌ width を取得できませんでした。
    pause
    exit /b 1
)
if "%HEIGHT%"=="" (
    echo ❌ height を取得できませんでした。
    pause
    exit /b 1
)

:: 验证是否为数字
for /f "delims=0123456789" %%i in ("%WIDTH%") do (
    echo ❌ width が数字ではありません: '%WIDTH%'
    pause
    exit /b 1
)
for /f "delims=0123456789" %%i in ("%HEIGHT%") do (
    echo ❌ height が数字ではありません: '%HEIGHT%'
    pause
    exit /b 1
)

:: 计算裁剪
for %%F in ("%INPUT%") do set "OUTPUT=%%~nF_中央.mp4"
set /a CROP_W=%WIDTH% - %LEFT% - %RIGHT%
set /a CROP_H=%HEIGHT% - %TOP% - %BOTTOM%

if %CROP_W% LEQ 0 (
    echo ❌ 幅が不足 (現在: %CROP_W%)
    pause
    exit /b 1
)
if %CROP_H% LEQ 0 (
    echo ❌ 高さが不足 (現在: %CROP_H%)
    pause
    exit /b 1
)

ffmpeg -y -i "%INPUT%" -vf "crop=%CROP_W%:%CROP_H%:%LEFT%:%TOP%" -c:a copy "%OUTPUT%"

if %errorlevel% equ 0 (
    echo ✅ 完了: %OUTPUT%
) else (
    echo ❌ ffmpeg エラー
)
pause
```

```html [video_cropper.html]
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>视频裁剪参数提取器</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      background: #f0f0f0;
    }
    .container {
      max-width: 1000px;
      margin: 0 auto;
    }
    h1 {
      text-align: center;
      color: #333;
    }
    #drop_zone {
      border: 3px dashed #ccc;
      border-radius: 10px;
      padding: 40px;
      text-align: center;
      margin: 20px 0;
      background: white;
      cursor: pointer;
    }
    #drop_zone.dragover {
      border-color: #4CAF50;
      background: #f9f9f9;
    }
    #video_container {
      position: relative;
      margin: 20px auto;
      /* 关键：不设背景色，避免黑边 */
      max-width: 1000px;
      overflow: visible;
    }
    #canvas {
      display: block;
      max-width: 100%;
      max-height: 700px; /* 限制最大高度 */
      /* 不设 height: auto，让浏览器自动等比缩放 */
    }
    #vertical_line, #horizontal_line {
      position: absolute;
      background: red;
      opacity: 0.8;
      z-index: 10;
      pointer-events: none;
    }
    #vertical_line {
      width: 2px;
      top: 0;
      bottom: 0;
    }
    #horizontal_line {
      height: 2px;
      left: 0;
      right: 0;
    }
    .controls {
      text-align: center;
      margin: 15px 0;
      background: white;
      padding: 15px;
      border-radius: 8px;
    }
    input[type="number"] {
      width: 80px;
      padding: 5px;
      margin: 0 10px;
    }
    button {
      padding: 8px 16px;
      background: #4CAF50;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    button:hover {
      background: #45a049;
    }
    #crop_params {
      background: #e8f5e9;
      padding: 15px;
      border-radius: 8px;
      margin-top: 20px;
      display: none;
    }
    .param-row {
      display: flex;
      justify-content: space-around;
      margin: 8px 0;
      font-weight: bold;
    }
    .param-value {
      background: white;
      padding: 5px 10px;
      border: 1px solid #4CAF50;
      border-radius: 4px;
      min-width: 60px;
      text-align: center;
    }
    .copy-btn {
      margin-top: 10px;
      padding: 6px 12px;
      font-size: 14px;
      background: #81C784;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>视频裁剪参数提取器</h1>
    
    <div id="drop_zone">
      <p>📁 将视频文件拖到此处</p>
      <p>或点击选择文件</p>
      <input type="file" id="file_input" accept="video/*" style="display:none;">
    </div>

    <div class="controls">
      <label>帧号: </label>
      <input type="number" id="frame_input" min="1" value="300">
      <button id="extract_btn">提取帧</button>
    </div>

    <div id="video_container" style="display:none;">
      <canvas id="canvas"></canvas>
      <div id="vertical_line"></div>
      <div id="horizontal_line"></div>
    </div>

    <div id="crop_params">
      <h3>裁剪参数（原始像素值）</h3>
      <div class="param-row">
        <div>LEFT (左): <span id="left_val" class="param-value">0</span> px</div>
        <div>RIGHT (右): <span id="right_val" class="param-value">0</span> px</div>
      </div>
      <div class="param-row">
        <div>TOP (上): <span id="top_val" class="param-value">0</span> px</div>
        <div>BOTTOM (下): <span id="bottom_val" class="param-value">0</span> px</div>
      </div>
      <div style="text-align:center;">
        <button class="copy-btn" onclick="copyCropParams()">📋 复制参数到剪贴板</button>
      </div>
    </div>
  </div>

  <script>
    const dropZone = document.getElementById('drop_zone');
    const fileInput = document.getElementById('file_input');
    const frameInput = document.getElementById('frame_input');
    const extractBtn = document.getElementById('extract_btn');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const videoContainer = document.getElementById('video_container');
    const vLine = document.getElementById('vertical_line');
    const hLine = document.getElementById('horizontal_line');
    const cropParams = document.getElementById('crop_params');
    
    const leftVal = document.getElementById('left_val');
    const rightVal = document.getElementById('right_val');
    const topVal = document.getElementById('top_val');
    const bottomVal = document.getElementById('bottom_val');

    let videoFile = null;
    let originalWidth = 0;
    let originalHeight = 0;

    // 拖拽 & 文件选择
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        videoFile = e.target.files[0];
        document.querySelector('#drop_zone p').textContent = `已加载: ${videoFile.name}`;
      }
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length && files[0].type.startsWith('video/')) {
        videoFile = files[0];
        document.querySelector('#drop_zone p').textContent = `已加载: ${videoFile.name}`;
      }
    });

    // ✅ 完整 extractFrame 函数（重点！）
    extractBtn.addEventListener('click', extractFrame);

    function extractFrame() {
      if (!videoFile) {
        alert('请先选择视频文件！');
        return;
      }

      const frameNumber = parseInt(frameInput.value) || 300;
      if (frameNumber < 1) {
        alert('帧号必须 ≥ 1');
        return;
      }

      const video = document.createElement('video');
      const url = URL.createObjectURL(videoFile);
      video.preload = 'metadata';
      video.src = url;

      video.onloadedmetadata = () => {
        const fps = 30;
        const time = (frameNumber - 1) / fps;
        if (time >= video.duration) {
          alert(`视频太短，无法提取第 ${frameNumber} 帧`);
          URL.revokeObjectURL(url);
          return;
        }

        video.currentTime = time;

        video.onseeked = () => {
          // 保存原始分辨率
          originalWidth = video.videoWidth;
          originalHeight = video.videoHeight;

          // 设置 canvas 为原始尺寸（内容不缩放）
          canvas.width = originalWidth;
          canvas.height = originalHeight;
          ctx.drawImage(video, 0, 0, originalWidth, originalHeight);

          // 显示容器
          videoContainer.style.display = 'block';

          // 初始化辅助线到中心
          updateLines(originalWidth / 2, originalHeight / 2);

          video.remove();
          URL.revokeObjectURL(url);
          cropParams.style.display = 'block';
        };
      };

      video.onerror = () => {
        alert('无法加载视频，请检查格式');
        URL.revokeObjectURL(url);
      };
    }

    // 鼠标拖动事件
    let isDragging = false;

    videoContainer.addEventListener('mousedown', () => {
      isDragging = true;
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging || !originalWidth) return;
      
      const canvasRect = canvas.getBoundingClientRect();
      const containerRect = videoContainer.getBoundingClientRect();

      // 计算鼠标在 canvas 内的原始坐标
      const x = (e.clientX - canvasRect.left) * (originalWidth / canvasRect.width);
      const y = (e.clientY - canvasRect.top) * (originalHeight / canvasRect.height);

      // 限制在原始尺寸内
      const clampedX = Math.max(0, Math.min(x, originalWidth));
      const clampedY = Math.max(0, Math.min(y, originalHeight));

      updateLines(clampedX, clampedY);
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });

    // ✅ 更新辅助线位置（关键！）
    function updateLines(x, y) {
      const canvasRect = canvas.getBoundingClientRect();
      const containerRect = videoContainer.getBoundingClientRect();

      // 计算辅助线在容器中的显示位置
      const displayX = canvasRect.left - containerRect.left + (x * canvasRect.width / originalWidth);
      const displayY = canvasRect.top - containerRect.top + (y * canvasRect.height / originalHeight);

      vLine.style.left = displayX + 'px';
      hLine.style.top = displayY + 'px';

      // 更新裁剪参数（原始像素值）
      const left = Math.round(x);
      const right = Math.round(originalWidth - x);
      const top = Math.round(y);
      const bottom = Math.round(originalHeight - y);

      leftVal.textContent = left;
      rightVal.textContent = right;
      topVal.textContent = top;
      bottomVal.textContent = bottom;
    }

    function copyCropParams() {
      const text = `TOP=${topVal.textContent}\nBOTTOM=${bottomVal.textContent}\nLEFT=${leftVal.textContent}\nRIGHT=${rightVal.textContent}`;
      navigator.clipboard.writeText(text).then(() => {
        alert('✅ 参数已复制！');
      }).catch(() => {
        alert('请手动复制下方数值');
      });
    }
  </script>
</body>
</html>
```
:::

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

## 时间轴
### 裁剪视频脚本
给定剪切时长和起始时间：
```shell
ffmpeg -i input.mp4 -ss 00:00:00 -t 100 -c copy part1.mp4
```

`cut_video_manual.bat`
```shell
@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ================== 配置区（只需改这4行）==================
set "INPUT=ビデオ2025_報告.mp4"
set "OUTPUT=切り抜き_セグメント1.mp4"
set "START=00:01:30"
set "END=00:03:45"
:: =========================================================

echo.
echo 切分视频：
echo   入力ファイル: %INPUT%
echo   出力ファイル: %OUTPUT%
echo   時間範囲: %START% ～ %END%
echo.

:: 执行 ffmpeg（-ss 放在 -i 前以加速，-c copy 不重新编码）
ffmpeg -y -ss %START% -to %END% -i "%INPUT%" -c copy "%OUTPUT%"

if %errorlevel% equ 0 (
    echo ✅ 切り抜き完了！
) else (
    echo ❌ エラーが発生しました。ファイル名や ffmpeg のインストールを確認してください。
)

pause
```

## 常见问题
手动安装torchvision时：‘AV_CODEC_CAP_INTRA_ONLY’ was not declared in this scope; did you mean ‘AV_CODEC_PROP_INTRA_ONLY’?

解决方法：卸载FFmpeg，然后安好了torchvision之后再安回去

