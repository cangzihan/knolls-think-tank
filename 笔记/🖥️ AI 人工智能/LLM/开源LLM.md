---
tags:
  - ChatGLM
  - Baichuan
  - Llama
  - LLM
---

# 开源LLM

一些好用的项目：
- https://github.com/mlc-ai/mlc-llm
- https://github.com/wangzhaode/mnn-llm

## ChatGLM
https://huggingface.co/THUDM/chatglm3-6b

### RK3588部署
参考【硬件】-【嵌入式】-【RK3588】中ChatGLM节


## Llama

### 部署
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

## Qwen
[Github](https://github.com/QwenLM/Qwen1.5)

Demo [Qwen1.5-72B-Chat](https://huggingface.co/spaces/Qwen/Qwen1.5-72B-Chat)

32B [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B) | [Qwen1.5-32B-Chat](https://huggingface.co/Qwen/Qwen1.5-32B-Chat) |

14B [Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B) | [Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat)

7B [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)

### 部署

32B显存占用65G

测试
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

## Qwen-VL

Qwen-VL-Max [Demo](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) | [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) | [Qwen-VL-Chat-Int4](https://huggingface.co/Qwen/Qwen-VL-Chat-Int4)
| [Paper](https://arxiv.org/abs/2308.12966)

- 显存占用：11.8G-28G
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

## Baichuan
https://www.baichuan-ai.com/home


[Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B)

[Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
