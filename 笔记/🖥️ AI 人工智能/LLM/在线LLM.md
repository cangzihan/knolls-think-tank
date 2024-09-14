---
tags:
  - ChatGPT
  - OpenAI
  - Baichuan
  - 通义千问
  - gemini
  - LLM
---

# 在线LLM
这里只考虑大语言模型，AIGC类模型不在这里考虑。

模型信息

|          | Model        | Web | API |
|----------|--------------|-----|-----|
| ChatGPT  | GPT-4        | ✔   | ✔   |
|          | GPT-4o       | ✔   | ✔   |
|          | GPT-4v       | ✔   |  ✔  |
|          | GPT-4o mini  | ✔   | ✔   |
|          | GPT-3.5      | -   | ✔   |
| Baichuan | Baichuan2    | ✔   | ✔   |
|          | Baichuan-NPC | ✔   | ✔   |
| 通义千问     | 通义千问         | ✔   | ✔   |


## API价格
统计日期 2024.7.25

大量请求可使用OpenAI Batch API 会有 <font color="green">-50%</font> 折扣。
<table>
  <tr>
    <th></th>
    <th>Model</th>
    <th>价格</th>
  </tr>
  <tr>
    <td rowspan="8">ChatGPT</td>
    <td>gpt-4o / gpt-4o-2024-05-13</td>
    <td>Input: $0.005 / 1K tokens <br>	Output: $0.015 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-4 </td>
    <td>Input: $0.03 / 1K tokens <br>	Output: $0.06 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-4-turbo / -2024-04-09 </td>
    <td>Input: $0.01 / 1K tokens <br>	Output: $0.03 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-4-32k</td>
    <td>Input: $0.06 / 1K tokens <br>	Output: $0.12 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-4o-mini / -2024-07-18 <font color="red">new</font> </td>
    <td>Input: $0.00015 / 1K tokens <br>	Output: $0.0006 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-3.5-turbo-0125</td>
    <td>Input: $0.0005 / 1K tokens <br>	Output: $0.0015 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-3.5-turbo-instruct</td>
    <td>Input: $0.0005 / 1K tokens <br>	Output: $0.0020 / 1K tokens</td>
  </tr>
  <tr>
    <td>gpt-4-vision-preview</td>
    <td>Input: $0.01 / 1K tokens <br>	Output: $0.003 / 1K tokens</td>
  </tr>
  <tr>
    <td rowspan="8">Baichuan</td>
    <td>Baichuan4</td>
    <td> ￥0.1 / 1K tokens</td>
  </tr>
  <tr>
    <td>Baichuan3-Turbo</td>
    <td> ￥0.012 / 1K tokens</td>
  </tr>
  <tr>
    <td>Baichuan2-Turbo</td>
    <td> ￥0.008 / 1K tokens</td>
  </tr>
  <tr>
    <td>Baichuan3-Turbo-128k</td>
    <td> ￥0.024 / 1K tokens</td>
  </tr>
  <tr>
    <td><s>Baichuan2-Turbo-192k</s></td>
    <td><s> ￥0.016 / 1K tokens</s></td>
  </tr>
  <tr>
    <td>Baichuan2-53B</td>
    <td> 0:00~8:00 ￥0.01 / 1K tokens <br>	8:00~24:00 ￥0.02 / 1K tokens</td>
  </tr>
  <tr>
    <td>Baichuan-NPC-Lite</td>
    <td> ￥0.0099 / 1K tokens</td>
  </tr>
  <tr>
    <td>Baichuan-NPC-Turbo</td>
    <td> ￥0.015 / 1K tokens</td>
  </tr>
  <tr>
    <td rowspan="4">通义千问</td>
    <td>qwen-turbo</td>
    <td>￥0.008/ 1K tokens</td>
  </tr>
  <tr>
    <td>qwen-plus</td>
    <td>￥0.02/ 1K tokens </td>
  </tr>
  <tr>
    <td>qwen-max</td>
    <td rowspan="2">￥0.12/ 1K tokens </td>
  </tr>
  <tr>
    <td>qwen-max-longcontext</td>
  </tr>
  <tr>
    <td rowspan="4">百度</td>
    <td>ERNIE-3.5-8K-0205</td>
    <td>Input: ￥0.024 / 1K tokens <br>	Output: ￥0.048 / 1K tokens</td>
  </tr>
  <tr>
    <td>ERNIE-3.5-4K-0205</td>
    <td>￥0.012 / 1K tokens </td>
  </tr>
  <tr>
    <td>ERNIE-Speed-8K</td>
    <td>Input: ￥0.004 / 1K tokens <br>	Output: ￥0.008 / 1K tokens</td>
  </tr>
  <tr>
    <td>ERNIE-4.0-8K</td>
    <td>￥50 / 420K tokens 1个月有效 </td>
  </tr>
</table>


## Poe(推荐)
Poe本身并不做大语言模型，但它汇总了大部分主流的LLM，对于刚接触大模型的新手也很有帮助。
https://poe.com/explore?category=Official

## Huggingface Chat
汇总了Huggingface上主流的LLM
https://huggingface.co/chat/

## ChatGPT
[Online](https://chat.openai.com/) | [API](https://platform.openai.com/settings/organization/billing/overview) | [AzureAPI](https://learn.microsoft.com/zh-cn/azure/ai-services/openai/) | [API定价](https://openai.com/api/pricing/)

### API
API最新价格：https://openai.com/pricing

微软的，都一个价格：https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/

API说明：https://platform.openai.com/docs/api-reference/chat/create

| 参数        | 类型               | 描述                                                                                                                                                                                   |
|-----------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `messages`  | list             | A list of messages comprising the conversation so far.                                                                                                                               |
| `model`  | string           | ID of the model to use.                                                                                                                                                              |
| `frequency_penalty`  | number or null   | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.        |
| `max_tokens`  | integer or null  | The maximum number of tokens that can be generated in the chat completion.                                                                                                           |
| `stream`  | boolean or null  |                                                                                                                                                                                      |
| `temperature`  | number or null     | What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. |


## 文心一言
[Online](https://yiyan.baidu.com/)

API用起来很繁琐：https://console.bce.baidu.com/qianfan/overview

## 通义千问
[API](https://dashscope.aliyun.com/)

API最新价格：https://dashscope.aliyun.com/

## 百川
[Online](https://www.baichuan-ai.com/chat?from=%2Fhome) |
[API](https://platform.baichuan-ai.com/docs/api)

### API
API最新价格：https://platform.baichuan-ai.com/price

```python
import os
os.environ["all_proxy"] = ""

import json
import requests


class Baichuan_turbo:
    def __init__(self):
        self.url = "https://api.baichuan-ai.com/v1/chat/completions"
        self.api_key = "XXXXXXXXXX"

    def call_llm(self, query, bc_model="Baichuan2-Turbo", stream=False, verbose=True):
        def end_sentence(sent):
            if sent and (sent.endswith("……") or
                         (sent.endswith("：") and len(sent) > 15) or
                         (sent.endswith("，") and len(sent) > 15) or
                         sent.endswith("。") or
                         sent.endswith("！") or
                         sent.endswith("？")):
                return True
            return False
        if not query:
            print('query is empty')
            return ''
        data = {
            "model": bc_model,
            "messages": query,
            "stream": stream  # 流式
        }

        json_data = json.dumps(data)

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

        response = requests.post(self.url, data=json_data, headers=headers, timeout=60, stream=True)

        if response.status_code == 200:
            if stream:
                collected_messages = ''
                if verbose:
                    print("请求成功！")
                    print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
                for line in response.iter_lines():
                    if line:
                        if verbose:
                            print(line.decode('utf-8'))
                        rcv_data = line[6:]
                        if b'{' in rcv_data:
                           # print(json.loads(rcv_data)['choices'][0]['delta']['content'])
                            chunk_content = json.loads(rcv_data)['choices'][0]['delta']['content']
                            collected_messages += chunk_content.replace("\n", "")

                            if end_sentence(collected_messages):
                                yield collected_messages
                                collected_messages = ''
                            elif '。' in collected_messages and len(collected_messages) > 15:
                                yield collected_messages.split('。')[0] + '。'
                                collected_messages = collected_messages[collected_messages.index('。')+1:]
                            elif '，' in collected_messages and len(collected_messages) > 20:
                                yield collected_messages.split('，')[0] + '，'
                                collected_messages = collected_messages[collected_messages.index('，')+1:]

                if len(collected_messages) > 0:
                    yield collected_messages
                yield "[Done]"
            else:
                answers = json.loads(list(response.iter_lines())[0])['choices'][0]['message']['content']
                return answers
        else:
            print("请求失败，状态码:", response.status_code)
            print("请求失败，body:", response.text)
            print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
```
