---
tags:
  - 语音
  - TTS
---

# ChatTTS

[Code](https://github.com/2noise/ChatTTS)

[相关资源汇总（推荐）](https://github.com/libukai/Awesome-ChatTTS) | [音色下载](https://huggingface.co/spaces/taa/ChatTTS_Speaker)

因为我不是做语音的，所以并不感兴趣它的具体原理，大概需要至少6G显存。

## Install

### Win/Mac
直接下载整合包（Win/Mac ）: https://www.bilibili.com/video/BV1pM4m1z7h9

### Linux

```shell
#git clone https://github.com/2noise/ChatTTS.git

conda create -n chat_tts python=3.10
conda activate chat_tts

#cd ChatTTS-main
#pip install .
#pip install -r requirements.txt
#可能还要重新安装一遍
# pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# pip3 install chattts-fork

pip install ChatTTS
pip install soundfile
```

将model文件存到一个文件夹

Python代码
::: code-group
```python [quickstart]
import ChatTTS
import soundfile

chat = ChatTTS.Chat()
chat.load_models(source='local', compile=False, local_path=【路径】) # Set to True for better performance  # [!code --]
chat.load(source='custom', compile=False, custom_path=【路径】) # Set to True for better performance # [!code ++]

texts = ["六王毕，四海一，赵爽是个大傻逼",]

wavs = chat.infer(texts, )

soundfile.write("output1.wav", wavs[0][0], 24000)
```

```python [advanced]
import ChatTTS
import torch
import soundfile

chat = ChatTTS.Chat()
chat.load_models(source='local', compile=False, local_path=【路径】) # Set to True for better performance  # [!code --]
chat.load(source='custom', compile=False, custom_path=【路径】) # Set to True for better performance # [!code ++]

texts = ["六王毕，四海一，赵爽是个大傻逼",]

spk = torch.load("seed_1397_restored_emb.pt")
params_infer_code = { # [!code --]
    'spk_emb': spk, # [!code --]
        # 'spk_emb': rand_spk, # add sampled speaker # [!code --]
  'temperature': 0.95, # using custom temperature # [!code --]
  'top_P': 0.7, # top P decode # [!code --]
  'top_K': 20, # top K decode # [!code --]
} # [!code --]
params_infer_code = ChatTTS.Chat.InferCodeParams( # [!code ++]
  spk_emb = spk, # [!code ++]
  temperature = 0.55, # [!code ++]
  top_P = 0.7, # [!code ++]
  top_K = 20, # top K decode # [!code ++]
) # [!code ++]

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_0][laugh_0][break_6]',
)

wavs = chat.infer(
  texts,
  params_refine_text=params_refine_text,
  params_infer_code=params_infer_code)

#torchaudio.save("output1.wav", torch.from_numpy(wavs[0]), 24000)
soundfile.write("output1.wav", wavs[0][0], 24000)
```

:::

命令行模式：
```shell
python3 examples/cmd/run.py "六王毕,四海一,赵爽是个大傻逼"
```
你可能需要修改`examples/cmd/run.py`的`chat.load_models`那行的模型地址


WebUI:
```shell
pip install gradio

python examples/web/webui.py --local_path 【路径】
```

## Usage

注意事项：文本中不要有`!`, `"`, `!`, `......`, `“`, `”`等符号

女声seed:
- 2, 13, 22, 29, 61, 65
- 8, 19, 26, 28, 32, 53, 66, 73
- 10（偏小孩）,18

男生seed:
1, **5**, 6, 7, 23, 63, 73, 76

商业Idea:
- (配合GPT4创作/直接下载现成的/翻译国外的)短篇恐怖故事

### 批量生成数据
```shell
import ChatTTS
from IPython.display import Audio
import torch
import torchaudio
import soundfile
import os

class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.state)

chat = ChatTTS.Chat()
chat.load(source='custom', compile=False, custom_path=【路径】) # Set to True for better performance

inputs_cn = """
 xl
""".replace('\n', '')

out_dir = "out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for audio_seed_input in range(10000):
    with TorchSeedContext(audio_seed_input):
        rand_spk = chat.sample_random_speaker()
    #print(rand_spk) # save it for later timbre recovery


    params_infer_code = ChatTTS.Chat.InferCodeParams(
      spk_emb = rand_spk, # add sampled speaker
      temperature = 0.8, # using custom temperature
      top_P = 0.7, # top P decode
      top_K = 20, # top K decode
    )

    params_refine_text = ChatTTS.Chat.RefineTextParams(
      prompt='[oral_0][laugh_0][break_4]',
    )
    audio_array_cn = chat.infer(inputs_cn, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

    soundfile.write(os.path.join(out_dir, "xl_t08_p07_k20_seed_%04d.wav"%audio_seed_input), audio_array_cn[0][0], 24000)

```


## Other
SeedTTS: https://bytedancespeech.github.io/seedtts_tech_report/#full-diffusion-samples

[Paper](https://arxiv.org/pdf/2406.02430)

### FunAudioLLM
https://fun-audio-llm.github.io/

语音生成
https://github.com/FunAudioLLM/CosyVoice

语音识别
https://github.com/FunAudioLLM/SenseVoice
