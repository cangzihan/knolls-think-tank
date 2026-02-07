
# vLLM

VLLM（全称 vLLM，有时也被解释为 virtual Large Language Model）是一个开源的、用于大语言模型（LLM）。它最初由加州大学伯克利分校的 Sky Computing Lab 开发。

其核心优势在于：
- 高吞吐量和内存效率：vLLM 能够高效地处理大量请求，并优化内存使用，从而降低成本。
- PagedAttention 技术：这是 vLLM 采用的一项关键技术，它通过更智能地管理注意力机制中的键值（KV）缓存，显著提高了内存利用率和吞吐量。
- 易于使用：它提供了一个简单易用的接口，方便开发者快速部署和提供 LLM 服务

| 工具 | 优势 | 劣势 |
|------|------|------|
| **vLLM** | 吞吐量极高、显存效率高、易用 | 仅支持 decoder-only 模型（不支持 encoder-decoder 如 T5） |
| **Hugging Face TGI** | 功能全面、支持多模型架构 | 吞吐量低于 vLLM |
| **TensorRT-LLM** | 极致性能（需编译） | 配置复杂，模型支持有限 |
| **DeepSpeed-Inference** | 微软生态集成好 | 部署复杂度高 |

## Install

Install
```shell
pip install vllm
```

## Usage
### OpenAI API
```shell
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-Instruct \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 8000

```
客户端
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # vLLM 不验证 key，可任意填写
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-Instruct",
    messages=[{"role": "user", "content": "你好！"}]
)
print(response.choices[0].message.content)

```

### python
```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="meta-llama/Llama-3-8b-Instruct")

# 设置生成参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

# 输入 prompt
prompts = ["Hello, how are you?", "Explain quantum computing in simple terms."]

# 批量推理
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

```

## 相关容器
### vllm-openai
Page: https://hub.docker.com/r/vllm/vllm-openai

::: code-group
```shell [Container start]
docker run --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --max-model-len 8192

```

```python [客户端]
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # vLLM 不验证 key，但需提供
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.choices[0].message.content)

```
:::
