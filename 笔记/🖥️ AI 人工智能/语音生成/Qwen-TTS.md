# Qwen-TTS
## Qwen3-TTS
[Model list](https://huggingface.co/collections/Qwen/qwen3-tts) | [Demo](https://huggingface.co/spaces/Qwen/Qwen3-TTS)

Qwen3-TTS可以用于定义自定义语音（音色，语气）和声音克隆。（但两个功能不能同时使用）

1. Voice CLone
[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
|
[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

2. Voice Design
[Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)

## Install
```shell
uv venv
source .venv/bin/activate
uv pip install qwen-tts accelerate
```
### Dockerfile Example

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y ffmpeg
RUN apt-get install -y sox

RUN pip install qwen-tts
RUN pip install accelerate

EXPOSE 8022

COPY . /workspace
ENV CUDA_VISIBLE_DEVICES=1
CMD ["python3", "main.py"]
```

## vLLM-Omni支持
[安装方法](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/)

参考代码：https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts
  