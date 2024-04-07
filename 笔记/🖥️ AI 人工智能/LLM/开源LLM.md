---
tags:
  - CHatGLM
  - Baichuan
  - Llama
  - LLM
---

# 开源LLM

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

```python
import eventlet
import time

eventlet.monkey_patch()
from eventlet import wsgi

from flask import Flask, jsonify, request, make_response
from flask_pymongo import PyMongo
from flask_cors import CORS, cross_origin
from transformers import AutoModel, LlamaForCausalLM, LlamaTokenizer, LlamaModel
from transformers.generation.utils import GenerationConfig
import uvicorn, json, datetime
import torch
import uuid
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

app.config["MONGO_URI"] = "mongodb://192.168.1.87/chat"
bc2_mongodb = PyMongo(app)

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
            "errmsg":"请求参数错误，ex:{'prompt':'hello','userid':'123','faq_type':0}",
        }
        ws.send(answer)
        ws.close()
        return 201

    faq_type = chat_req['faq_type']
    prompt = chat_req['prompt']
    query_faq = None
    mongodb = bc2_mongodb
    if faq_type == 0:
        query_faq = prompt

    userid = chat_req["userid"]
    qc = mongodb.db.user.count_documents({"uuid":userid})
    if qc:
        history = mongodb.db.user.find_one({"uuid":userid})['history']
    else:
        history = []
        user = {
            "uuid":userid,
            "history":history
        }
        mongodb.db.user.insert_one(user)

    print(history)
    history.append({"role":"user", "content":query_faq})
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
        "status":1
    }
    history = history if len(history)<5 else history[-5:]
    history.append({"role":"user", "content":prompt})
    history.append({'role':'assisant','content':response})
    print('history-after', history)
    mongodb.db.user.update_one({'uuid':userid},{'$set':{'history':history}})
    ws.send(answear)
    ws.close()
    return 200

@app.route("/chat", methods=['POST'])
def dialog():
    if request.content_type.startswith('application/json'):
        if 'faq_type' not in request.json:
            resp = {
                "errmsg":"no param 'faq_type'",
                "status":201
            }
            return resp
    faq_type = request.json['faq_type']
    prompt = request.json['prompt']
    msg = []
    query_faq = None
    mongodb = bc2_mongodb
    if faq_type == 0:
        query_faq = prompt

    # print(query_faq)

    userid = request.json['userid']
    if userid == "000":
        # msg.append({"role":"user", "content":query_faq})
        # response = model.chat(tokenizer, msg)

        #response = model.chat(tokenizer, query_faq)
        response = chat_API(query_faq)
        answear = {
            "userid":userid,
            "response":response,
            "status":faq_type
        }
    elif userid == "111":
        msg = [{"role":"user", "content":query_faq}]
       # response = model.chat(tokenizer, msg)
        response = chat_API(msg)

        answear = {
            "userid":userid,
            "response":response,
            "status":faq_type
        }
    else:
        qc = mongodb.db.user.count_documents({"uuid":userid})
        if qc:
            history = mongodb.db.user.find_one({"uuid":userid})['history']
        else:
            history = []
            user = {
                "uuid":userid,
                "history":history
            }
            mongodb.db.user.insert_one(user)

        history.append({"role":"user", "content":query_faq})

        response = model.chat(tokenizer, history)
        answear = {
            "userid":userid,
            "response":response,
            "status":faq_type
        }
        history = history if len(history)<5 else history[-5:]
        history.append({"role":"user", "content":prompt})
        history.append({'role':'assisant','content':response})
        # print('history-after', history)
        mongodb.db.user.update_one({'uuid':userid},{'$set':{'history':history}})
        # print(response)
        torch_gc()
    return answear

if __name__ == '__main__':
    wsgi.server(eventlet.listen(('0.0.0.0',3000)), app)

```

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
