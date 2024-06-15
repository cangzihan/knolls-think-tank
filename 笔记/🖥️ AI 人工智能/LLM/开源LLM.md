---
tags:
  - ChatGLM
  - Baichuan
  - Llama
  - 通义千问
  - LLM
---

# 开源LLM

## 实用工具
一些好用的项目：
- https://github.com/mlc-ai/mlc-llm
- https://github.com/wangzhaode/mnn-llm
- https://lmstudio.ai/
- https://github.com/ollama/ollama

### FastChat
[Github](https://github.com/lm-sys/FastChat?tab=readme-ov-file)

FastChat 是一个强大的平台，可以帮助你训练、部署和评估基于大型语言模型的聊天机器人。它提供了训练和评估代码，以及分布式多模型服务系统，使你能够轻松地构建和管理自己的聊天机器人项目。

#### Install
```shell
pip3 install "fschat[model_worker,webui]"
```

解决stream接口问题

- 方法1

不推荐，会和新版本gradio冲突
```shell
pip install pydantic==1.10.14
```

- 方法2

在API的输出中，找到报错的openai路径，如报错为
```shell
File "/home/XXX/.conda/envs/py310_really_python_3_10/lib/python3.10/site-packages/fastchat/serve/openai_api_server.py", line 942, in <module>
2024-05-13 10:52:07 | ERROR | stderr |     uvicorn.run(app, host=args.host, port=args.port, log_level="info")
```
就修改文件`/home/XXX/.conda/envs/py310_really_python_3_10/lib/python3.10/site-packages/fastchat/serve/openai_api_server.py`的3个地方

`line 506`:
```python
    for i in range(n):
        # First chunk with role
        choice_data = ChatCompletionResponseStreamChoice(
            index=i,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n" # [!code --]
        yield f"data: {json.dumps(chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n" # [!code ++]

        previous_text = ""
```

`line 536`和`line 539`:
```python
            if delta_text is None:
                if content.get("finish_reason", None) is not None:
                    finish_stream_events.append(chunk)
                continue
            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n" # [!code --]
            yield f"data: {json.dumps(chunk.dict(exclude_unset=True), ensure_ascii=False)}\n\n" # [!code ++]
    # There is not "content" field in the last delta message, so exclude_none to exclude field "content".
    for finish_chunk in finish_stream_events:
        yield f"data: {finish_chunk.json(exclude_none=True, ensure_ascii=False)}\n\n" # [!code --]
        yield f"data: {json.dumps(finish_chunk.dict(exclude_none=True), ensure_ascii=False)}\n\n" # [!code ++]
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
```

#### 部署LLM
需要同时开启多个终端
```shell
# Controller
python3 -m fastchat.serve.controller
# Worker
# worker 0
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5 --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path lmsys/fastchat-t5-3b-v1.0 --controller http://localhost:21001 --port 31001 --worker http://localhost:31001
# Web UI
python3 -m fastchat.serve.gradio_web_server
```

https://rudeigerc.dev/posts/llm-inference-with-fastchat/



## 基本环境搭建

很多LLM需要的环境都是类似的，这里默认在说电脑/服务器端、假设已经装好了GPU驱动、Cuda、Torch。
```shell
pip install transformers
```

## Baichuan
Baichuan 2 是百川智能推出的开源LLM，所有版本不仅对学术研究完全开放，开发者也仅需邮件申请并获得官方商用许可后，即可以免费商用。

https://www.baichuan-ai.com/home

网页端免费[ChatBot](https://www.baichuan-ai.com/chat?from=%2Fhome)

7B [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)

13B [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)

其余Baichuan2模型均不推荐

## ChatGLM
ChatGLM 是一个开源的、支持中英双语问答的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构。

ChatGLM 3
https://huggingface.co/THUDM/chatglm3-6b

### RK3588部署
参考【硬件】-【嵌入式】-【RK3588】中ChatGLM节


## Llama

### Llama2
#### 1.下载
需要下载**hf版本**的Llama，如[Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

这里以`Llama-2-13b-chat-hf`为例，实际上不需要下载全部文件，如果时间紧可以下载如下：
```
config.json
generation_config.json
gitattributes
LICENSE.txt
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
pytorch_model.bin.index.json
README.md
Responsible-Use-Guide.pdf
special_tokens_map.json
tokenizer_config.json
tokenizer.json
tokenizer.model
USE_POLICY.md
```

#### 2.调用
将所有文件下载下来放到一个文件夹里，命名为`Llama-2-13b-chat-hf`，然后在文件夹同级目录下创建Python脚本
```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(
        "Llama-2-13b-chat-hf",
        device_map="auto",
        torch_dtype=torch.float16
        )

tokenizer = AutoTokenizer.from_pretrained("Llama-2-13b-chat-hf")

while True:
    prompt = input(">>")
    if prompt == "exit":
        break
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    ans = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(ans)
```

#### 3.部署

- 服务端
```shell
pip install protobuf
```

创建`chat.py`和`app.py`:

::: code-group
```python [chat]                                                                                                                                           23,0-1       顶端
from transformers import AutoTokenizer
import transformers
import torch

model = "Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def chat_API(messages):
    if messages[0]['role'] != "system" or len(messages)==1:
        system_msg = "You are AI assistant"
        dialogue = f"<s>[INST] <<SYS>> " \
                   f"{system_msg} " \
                   f"<</SYS>> " \
                f"{messages[0]['content']} [/INST]"
        dialogue += "\n"

        sequences = pipeline(
            f'{dialogue}\n',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )

        prompt_length = len(dialogue)
        prompt_text = sequences[0]['generated_text'][:prompt_length]
        generated_part = sequences[0]['generated_text'][prompt_length:]
        generated_part = generated_part.strip()

    else:
        system_msg = messages[0]['content']
        dialogue = f"<s>[INST] <<SYS>> " \
                   f"{system_msg} " \
                   f"<</SYS>> " \
                f"{messages[1]['content']} [/INST]"
        dialogue += "\n"

        sequences = pipeline(
            f'{dialogue}\n',
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2048,
        )

        prompt_length = len(dialogue)
        prompt_text = sequences[0]['generated_text'][:prompt_length]
        generated_part = sequences[0]['generated_text'][prompt_length:]
        generated_part = generated_part.strip()

    return generated_part

message = [
        {'role': "system", 'content': "你是1个AI助手"},
        {'role': "user", 'content': "帮我记住一个词，Button"},
        {'role': "assistant", 'content': "帮我记住一个词，Button"},
        {'role': "user", 'content': "我要你记住的词是什么"}]


message1 = [{'role': "system", 'content': "你是1个AI助手, 请在所有对话之前说[你好,]"}, {'role': "user", 'content': "What is the captial of Beijing?"}]
rec = chat_API(message1)

print(message1)
print(rec)
```

```python [app]
import eventlet

eventlet.monkey_patch()
from eventlet import wsgi

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import torch
from flask_sock import Sock, Server

from chat import chat_API

DEVICE = "cuda"
# DEVICE_IDS = ["0", "1", "2"]
DEVICE_IDS = ["0"]


def torch_gc():
    if torch.cuda.is_available():
        for device_id in DEVICE_IDS:
            cuda_device = f"{DEVICE}:{device_id}"
            with torch.cuda.device(cuda_device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


app = Flask(__name__)
sock = Sock(app)
app.config['JSON_AS_ASCII'] = False
CORS(app)


def end_sentence(sent):
    if sent and (
        sent.endswith("......") or
        sent.endswith("。") or
        sent.endswith("！") or
        sent.endswith("？")
    ):
        return True
    return False


@sock.route("/chat_stream")
def dialog_stream(ws:Server):
    msg = ws.receive()
    chat_req = eval(msg)
    if 'prompt' not in chat_req or 'userid' not in chat_req or 'faq_type' not in chat_req:
        answer = {
            "errmsg": "请求参数错误，ex:{'prompt':'hello','userid':'123','faq_type':0}",
        }
        ws.send(answer)
        ws.close()
        return 201

    faq_type = chat_req['faq_type']
    prompt = chat_req['prompt']
    query_faq = None
    if faq_type == 0:
        query_faq = prompt

    userid = chat_req["userid"]
    history = []

    print(history)
    history.append({"role": "user", "content": query_faq})
    history_reponse = ''

    for response in model.chat(tokenizer, history, stream=True):
        if end_sentence(response):
            response = response.replace('\n',"")
            current_response = response.replace(history_reponse, '')
            if current_response:
                history_reponse += current_response
                answear = {
                    "userid": userid,
                    "response": current_response,
                    "status": 0
                }
                print(current_response)
                ws.send(answear)
    torch_gc()
    answear = {
        "userid": userid,
        "response": "end",
        "status": 1
    }
    history = history if len(history)<5 else history[-5:]
    history.append({"role": "user", "content": prompt})
    history.append({'role': 'assisant','content': response})
    print('history-after', history)
    ws.send(answear)
    ws.close()
    return 200


@app.route("/chat", methods=['POST'])
def dialog():
    if request.content_type.startswith('application/json'):
        if 'faq_type' not in request.json:
            resp = {
                "errmsg": "no param 'faq_type'",
                "status": 201
            }
            return resp
    faq_type = request.json['faq_type']
    prompt = request.json['prompt']
    query_faq = None
    if faq_type == 0:
        query_faq = prompt

    userid = request.json['userid']
    if userid == "000":
        # msg.append({"role":"user", "content":query_faq})
        # response = model.chat(tokenizer, msg)

        #response = model.chat(tokenizer, query_faq)
        response = chat_API(query_faq)
        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
    elif userid == "111":
        msg = [{"role": "user", "content": query_faq}]
       # response = model.chat(tokenizer, msg)
        response = chat_API(msg)

        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
    else:
        history = []
        history.append({"role": "user", "content": query_faq})

        response = model.chat(tokenizer, history)
        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
        torch_gc()
    return answear


if __name__ == '__main__':
    wsgi.server(eventlet.listen(('0.0.0.0', 3100)), app)
```
:::

- 调用
```python
    query = [{'role': "system", 'content': system},
             {'role': "user", 'content': prompt}]
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    payload = {'prompt': query, 'userid': "000", 'faq_type': 0}
    request = urllib.request.Request(【ip地址】, json.dumps(payload).encode("utf-8"), headers=headers)
    while True:
        try:
          response = urllib.request.urlopen(request, timeout=300)
          break
        except:
            print("Request错误")
            time.sleep(1)
```

### Llama3
[Project](https://huggingface.co/blog/llama3)

[Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

一些模型说明：
- 训练环境：24000个GPU


## Qwen
**通义千问（Qwen）** 是阿里云研发的基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。

[Github](https://github.com/QwenLM/Qwen1.5) | [Paper](https://arxiv.org/abs/2309.16609)

Demo [Qwen1.5-72B-Chat](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat) | [Qwen1.5-110B-Chat](https://huggingface.co/spaces/Qwen/Qwen1.5-110B-Chat-demo)

32B [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B) | [Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat) |

14B [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B) | [Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat)

7B [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

[CodeQwen1.5-7b-Chat](https://huggingface.co/spaces/Qwen/CodeQwen1.5-7b-Chat-demo)


### Quick Start
#### 1. 命令行
使用[FastChat](#fastchat)
```shell
python3 -m fastchat.serve.cli --model-path Qwen1.5-32B-Chat
```

####  2. Python脚本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-32B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-32B-Chat")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```


### 部署

32B显存占用65G

#### 1. Flask版
::: code-group
```python [chat]
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen1.5-32B-Chat",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen1.5-32B-Chat")

def chat_API(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

message = [
        {'role': "system", 'content': "你是1个AI助手"},
        {'role': "user", 'content': "帮我记住一个词，Button"},
        {'role': "assistant", 'content': "帮我记住一个词，Button"},
        {'role': "user", 'content': "我要你记住的词是什么"}]


message1 = [{'role': "system", 'content': "你是1个AI助手, 请在所有对话之前说'你好,'"}, {'role': "user", 'content': "What is the captial of China?"}]
rec = chat_API(message1)

print(message1)
print(rec)
```

```python [app]
import eventlet

eventlet.monkey_patch()
from eventlet import wsgi

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS, cross_origin
import torch
from flask_sock import Sock, Server

from chat import chat_API

DEVICE = "cuda"
# DEVICE_IDS = ["0", "1", "2"]
DEVICE_IDS = ["0"]


def torch_gc():
    if torch.cuda.is_available():
        for device_id in DEVICE_IDS:
            cuda_device = f"{DEVICE}:{device_id}"
            with torch.cuda.device(cuda_device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


app = Flask(__name__)
sock = Sock(app)
app.config['JSON_AS_ASCII'] = False
CORS(app)


def end_sentence(sent):
    if sent and (
        sent.endswith("......") or
        sent.endswith("。") or
        sent.endswith("！") or
        sent.endswith("？")
    ):
        return True
    return False


@sock.route("/chat_stream")
def dialog_stream(ws:Server):
    msg = ws.receive()
    chat_req = eval(msg)
    if 'prompt' not in chat_req or 'userid' not in chat_req or 'faq_type' not in chat_req:
        answer = {
            "errmsg": "请求参数错误，ex:{'prompt':'hello','userid':'123','faq_type':0}",
        }
        ws.send(answer)
        ws.close()
        return 201

    faq_type = chat_req['faq_type']
    prompt = chat_req['prompt']
    query_faq = None
    if faq_type == 0:
        query_faq = prompt

    userid = chat_req["userid"]
    history = []

    print(history)
    history.append({"role": "user", "content": query_faq})
    history_reponse = ''

    for response in model.chat(tokenizer, history, stream=True):
        if end_sentence(response):
            response = response.replace('\n',"")
            current_response = response.replace(history_reponse, '')
            if current_response:
                history_reponse += current_response
                answear = {
                    "userid": userid,
                    "response": current_response,
                    "status": 0
                }
                print(current_response)
                ws.send(answear)
    torch_gc()
    answear = {
        "userid": userid,
        "response": "end",
        "status": 1
    }
    history = history if len(history)<5 else history[-5:]
    history.append({"role": "user", "content": prompt})
    history.append({'role': 'assisant','content': response})
    print('history-after', history)
    ws.send(answear)
    ws.close()
    return 200


@app.route("/chat", methods=['POST'])
def dialog():
    if request.content_type.startswith('application/json'):
        if 'faq_type' not in request.json:
            resp = {
                "errmsg": "no param 'faq_type'",
                "status": 201
            }
            return resp
    faq_type = request.json['faq_type']
    prompt = request.json['prompt']
    query_faq = None
    if faq_type == 0:
        query_faq = prompt

    userid = request.json['userid']
    if userid == "000":
        # msg.append({"role":"user", "content":query_faq})
        # response = model.chat(tokenizer, msg)

        #response = model.chat(tokenizer, query_faq)
        response = chat_API(query_faq)
        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
    elif userid == "111":
        msg = [{"role": "user", "content": query_faq}]
       # response = model.chat(tokenizer, msg)
        response = chat_API(msg)

        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
    else:
        history = []
        history.append({"role": "user", "content": query_faq})

        response = model.chat(tokenizer, history)
        answear = {
            "userid": userid,
            "response": response,
            "status": faq_type
        }
        torch_gc()
    return answear


if __name__ == '__main__':
    wsgi.server(eventlet.listen(('0.0.0.0', 3100)), app)
```
:::

#### 2. FastChat版

http://felixzhao.cn/Articles/article/71

开三个终端
```shell
# 第一个终端
python3 -m fastchat.serve.controller

# 第二个终端
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path Qwen1.5-32B-Chat --controller http://localhost:21001 --port 31000 --worker http://localhost:31000

# 第三个终端
## Gradio Web
python3 -m fastchat.serve.gradio_web_server

## API
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0
```

客户端：
```python
from openai import OpenAI

class FastChatAPI:
    def __init__(self, model, base_url=None, api_key="Empty"):
        if "gpt" not in model:
            # 使用Local LLM
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            # 使用GPT
            self.client = OpenAI()
        self.model = model

    def completion(self, query, verbose=True):
        """Create a completion."""
        return self.completion_history(query, verbose, None)
       # response = self.client.completions.create(
       #     model=["Qwen1.5-32B-Chat"][0],
       #     prompt=query,
       #     max_tokens=768
       # )

       # return response.choices[0].text

    def completion_history(self, query, verbose=True, history=None):
        """Create a completion."""
        if history is None:
            msg = [
                {'role': 'system', 'content': f'你是一个AI助手，请回答.'},
                {'role': 'user', 'content': query},
            ]
        else:
            msg = history + [{'role': 'user', 'content': query}]
            # print(msg)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg,
        )

        if verbose:
            print("使用的tokens：", response.usage.total_tokens)
            print("问：", query)
            print("回答：", response.choices[0].message.content)

        new_history = msg + [{"role": "assistant", "content": response.choices[0].message.content}]
        return new_history

    def completion_stream(self, query):
        return self.completion_stream_history(query, None)

    def completion_stream_history(self, query, history=None):
        def end_sentence(sent):
            if sent and (sent.endswith("……") or
                         sent.endswith("：") and len(sent) > 15 or
                         sent.endswith("，") and len(sent) > 15 or
                         sent.endswith("。") or
                         sent.endswith("！") or
                         sent.endswith("？")):
                return True
            return False

        if history is None:
            msg = [
                {'role': 'system', 'content': f'你是一个AI助手，请回答.'},
                {'role': 'user', 'content': query},
            ]
        else:
            msg = history + [{'role': 'user', 'content': query}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg,
            temperature=0,
            stream=True
        )
        collected_messages = ''
        start_time = time.time()
        for chunk in response:
            chunk_time = time.time() - start_time  # calculate the time delay of the chunk
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                collected_messages += chunk_content.replace("\n", "")
                if end_sentence(collected_messages):
                    print(f"Message received {chunk_time:.2f} seconds after request: {collected_messages}")
                    collected_messages = ''
        print(f"Full response received {chunk_time:.2f} seconds after request")

        return collected_messages
```

【报错】AttributeError: module 'asyncio' has no attribute 'to_thread'：

什么？你还在用`Python 3.9`之前的版本，真的是反清复明，开历史倒车！[解决方案](https://stackoverflow.com/questions/68523752/python-module-asyncio-has-no-attribute-to-thread)

 【报错】 流式API问题

[参考安装FastChat](#fastchat)

## Qwen-VL

Qwen-VL-Max [Demo🤖](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) | [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) | [Qwen-VL-Chat-Int4](https://huggingface.co/Qwen/Qwen-VL-Chat-Int4)
| [Paper](https://arxiv.org/abs/2308.12966)

- 显存占用：11.8G-28G (实测约为21G)
- token: 32768

一个基于Qwen API的ComfyUI节点：https://github.com/ZHO-ZHO-ZHO/ComfyUI-Qwen-VL-API?tab=readme-ov-file
- 首先输入一个图像，然后由Qwen-VL"describe the image"，然后将文字描述通过CLIP传给SD，实现image-to-image

### 部署
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

# 2nd dialogue turn
response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
print(response)
# <ref>击掌</ref><box>(517,508),(589,611)</box>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('1.jpg')
else:
  print("no box")
```

#### 输入
注意到原工程的中`tokenization_qwen.py`中
```python
    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}: '
                text += self.image_start_tag + ele['image'] + self.image_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text
```
可只除了`'image'`和`'text'`之外，还支持第三种输入方式，那就是`'box'`。对话中的检测框可以表示为`<box>(x1,y1),(x2,y2)</box>`，
其中 `(x1, y1)` 和`(x2, y2)`分别对应左上角和右下角的坐标，并且被归一化到`[0, 1000)`的范围内. 检测框对应的文本描述也可以通过`<ref>text_caption</ref>`表示。

第三种任务示例
```python
# ......可接上一个代码

# bbox输入
query = tokenizer.from_list_format([
    {'image': '00260-70362828.png'},
    {'text': '检测框中的人是站着还是坐着?'},
    {'box': [(272,271,526,957)]}
])
print("query:", query)

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
image = tokenizer.draw_bbox_on_latest_picture("<box>(272,271),(526,957)</box>", history)
if image:
  image.save('2.jpg')
else:
  print("no box")
```

服务(需要创建一个空文件夹命名为`data`)

::: code-group
```python [service]
from flask import Flask, request, jsonify
import time
from flask_sock import Sock, Server
from flask_cors import CORS
from eventlet import wsgi
import eventlet
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

torch.manual_seed(1234)

# Load tokenizer and model
# QianWen_VL 7B
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

app = Flask(__name__)
sock = Sock(app)
app.config['JSON_AS_ASCII'] = False
CORS(app)
img_path = "data/temp.jpg"

@app.route('/qianwen_vl', methods=['POST'])
def qianwen_vl():
    image_file = request.files['image']
    save_jpg(image_file, img_path)
    #prompt = request.form['prompt']
    print(json.loads(request.form.to_dict()['content']))
    query_list = [{'image' :img_path}] + json.loads(request.form.to_dict()['content'])
    print(query_list)
    query = tokenizer.from_list_format(query_list)
    print(query)
   # query = f'<img>{img_path}</img>{prompt}'
    start_time = time.time()
    response, _ = model.chat(tokenizer, query=query, history=None)
    end_time = time.time()

    return jsonify({'response': response, 'running_time': end_time - start_time})

def save_jpg(image_file, file_name):
    image = Image.open(image_file.stream)
    image.save(file_name, "JPEG")

if __name__ == '__main__':
    wsgi.server(eventlet.listen(('0.0.0.0', 5005)), app)
```

```python [API]
import cv2
import json
import requests

# 定义 Flask 服务器的地址
server_url = 'http://192.168.xxx.xxx:5005/qianwen_vl'


def query_qwen_vl(data):
    image_path = data[0]["image"]
    content = data[1:]
    # 转成字符串再发送给服务端
    data_send = {'content': json.dumps(content, ensure_ascii=False)}
    #print(data_send)
    # 打开并读取图像文件
    with open(image_path, 'rb') as f:
        # 构造请求数据
        files = {'image': f}

        try:
            # 发送 POST 请求
            response = requests.post(server_url, files=files, data=data_send)

            # 检查响应状态码
            if response.status_code == 200:
                # 解析 JSON 响应
                response_data = response.json()
                print("问：", data)
                print("对话响应:", response_data['response'])
                print("运行时间:", response_data['running_time'])
            else:
                print("请求失败:", response.status_code)
        except requests.RequestException as e:
            print("请求错误:", e)
    return response_data['response']


def extract_bbox(text):
    current_text = text[text.index('<box>'):]

    bboxs = []
    while '<box>' in current_text:
        current_text = current_text[current_text.index('<box>')+5:]
        current_text_list = eval("[%s]" % current_text[:current_text.index('</box>')])
        bboxs.append([current_text_list[0][0], current_text_list[0][1], current_text_list[1][0], current_text_list[1][1]])

    return bboxs


def draw_bbox_cv(img, box, text=None):
    thickness = 3  # 框的厚度
    # 画矩形框
    img1 = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), thickness)
    if text is not None:
        img1 = cv2.putText(img1, text, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return img1


def test1():
    # 图像文件路径
    image_file = '00346-4016997344.png'
    data = [{"image": image_file}, {"text": "图中是什么家具？"}]
    query_qwen_vl(data)


def test2():
    # 图像文件路径
    image_file = '00260-70362828.png'
    data = [{"image": image_file}, {"text": "输出人的检测框"}]
    res = query_qwen_vl(data)
    bboxs = extract_bbox(res)

    # 画图
    image = cv2.imread(image_file)
    for i, bbox in enumerate(bboxs):
        image = draw_bbox_cv(image, bbox, str(i))
    cv2.imshow("Image", image)
    cv2.waitKey()


def test3():
    test_bbox = (267,269,526,958)
    # 图像文件路径
    image_file = '00260-70362828.png'
    data = [{"image": image_file}, {"text": "检测框中的人穿的什么衣服？"}, {"box": [test_bbox]}]
    query_qwen_vl(data)

    # 画图
    image = cv2.imread(image_file)
    image = draw_bbox_cv(image, test_bbox)
    cv2.imshow("Image", image)
    cv2.waitKey()


if __name__ == "__main__":
    test3()
```
:::

```shell
pip install tiktoken matplotlib
```

## GLM4
[glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)

## 其他
XXXXXXX
