---
tags:
  - 大模型
  - LLM
---

# langchain

## 安装
```shell
pip install langchain
pip install -U langchain-community
pip install -U langchain-openai
```

## prompt模板
```shell
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("你是一个起名大师,请模仿示例起3个{county}名字,比如男孩经常被叫做{boy},女孩经常被叫做{girl}")
message = prompt.format(county="中国特色的", boy="狗蛋", girl="翠花")
print(message)
```
