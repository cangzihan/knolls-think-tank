
# ASR

## whisper

支持中文，低配置可以处理长音频

[Demo](https://huggingface.co/spaces/openai/whisper) | [Paper](https://arxiv.org/abs/2212.04356)

[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai\whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe("audio_file.mp3")
print(result["text"])
```

