---
tags:
  - Prompt
  - 大模型
---


# RAG

Retrieval-Augmented Generation是指对大型语言模型输出进行优化，使其能够在生成响应之前引用训练数据来源之外的权威知识库。
通用语言模型通过微调就可以完成几类常见任务，比如分析情绪和识别命名实体。这些任务不需要额外的背景知识就可以完成。

要完成更复杂和知识密集型的任务，可以基于语言模型构建一个系统，访问外部知识源来做到。这样的实现与事实更加一性，生成的答案更可靠，还有助于缓解“幻觉”问题。

[Paper](https://arxiv.org/abs/2005.11401)

## 背景
传统上，要让神经网络适应特定领域或私有信息，人们通常会对模型进行微调。这种方法确实有效，但同时也耗费大量计算资源、成本高昂，且需要丰富的技术知识，因此在快速适应信息变化方面并不灵活。

用一个简单的比喻来说， RAG 对大语言模型（Large Language Model，LLM）的作用，就像开卷考试对学生一样。在开卷考试中，学生可以带着参考资料进场，比如教科书或笔记，用来查找解答问题所需的相关信息。开卷考试的核心在于考察学生的推理能力，而非对具体信息的记忆能力。

## 原理
RAG案例-离线流程

```text
       文档解C               切片
PDF ----------> MarkDown ---------> 切片1   切片2
                                     |       |
                                     |       |
                                     V       V
                                    向量1   向量2
```

RAG案例-在线流程

```text
Q: 如何搭配食物才能满足高温作业的需求？
            |
            | 首次检索
            V                                                      最终总结
       切片1   切片2                                  XXXXXXXXXXXX --------->  XXXXXXXXXXXX
            |                                             ^
            | 首次总结                              二次总结 |
            V                                             |
       XXXXXXXXX                                     切片1   切片2
            |                                             ^
            | 推理                                         |
            V                      生成子问题               |
还缺少富含钾、维生素C的蔬菜水果具体名称 -----------> 哪些蔬菜和水果富含锌、维生素C？
```

## Weaviate

Weaviate is an open-source vector database. It enables you to store data objects and vector embeddings and query them based on similarity measures.

### Prepare
```shell
pip install langchain openai weaviate-client
```

## Embedding
### BCEmbedding
https://github.com/netease-youdao/BCEmbedding

## langchain
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它使得应用程序能够：

- 具有上下文感知能力：将语言模型连接到上下文来源（提示指令，少量的示例，需要回应的内容等）
- 具有推理能力：依赖语言模型进行推理（根据提供的上下文如何回答，采取什么行动等）

## QAnything
[QAnything本地知识库问答系统：基于检索增强生成式应用（RAG）两阶段检索、支持海量数据、跨语种问答](https://www.cnblogs.com/ting1/p/17979702)

项目地址：https://github.com/netease-youdao/QAnything

### 安装

#### Quick start
```shell
git clone https://github.com/netease-youdao/QAnything.git
cd QAnything
# 使用GPU4
git checkout remotes/origin/master # 新版本弃用了很多命令导致有问题
sudo bash ./run.sh -c local -i 4 -b default

# 报错：Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
sudo nala install -y nvidia-container-toolkit
sudo systemctl restart docker
```

- `-i`控制device_id
- `-c`设定`llm_api`
- `-b`为`default`时，且`llm_api`不为`cloud`时，使用7B模型。（`-b`默认为`default`，显存不够可以选`hf`）

网页端：http://[ip地址]:8777/qanything/#/home

[接口文档](https://github.com/netease-youdao/QAnything/blob/master/docs/API.md)

### 代码分C

查看容器日志
```shell
sudo docker logs qanything-container-local
```

进入容器内部
```shell
sudo docker exec -it qanything-container-local bash
```

退出
```shell
exit
```

对话函数在`QAnything/qanything_kernel/qanything_server/handler.py`中

`QAnything/qanything_kernel/core/local_doc_qa.py`中`LocalDocQA`类的`get_knowledge_based_answer`

进一步可以看到他们chat的prompt模板,位于`QAnything/qanything_kernel/configs/model_config.py`：
```python
PROMPT_TEMPLATE = """参考信息：
{context}
---
我的问题或指令：
{question}
---
请根据上述参考信息回答我的问题或回复我的指令。前面的参考信息可能有用，也可能没用，你需要从我给出的参考信息中选出与我的问题最相关的那些，来为你的回答提供依据。回答一定要忠于原文，简洁但不丢信息，不要胡乱编造。我的问题或指令是什么语种，你就用什么语种回复,
你的回复："""
```

检索的核心代码为：
```python
        source_documents = self.get_source_documents(retrieval_queries, milvus_kb)

        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        if rerank and len(retrieval_documents) > 1:
            debug_logger.info(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=PROMPT_TEMPLATE)
```

所有的API描述在`qanything_kernel/qanything_server/sanic_api.py`里，在`line 79`中含有问答接口：
```python
app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])  # tags=["问答接口"]
```
可知，问答接口执行了一个`local_doc_chat()`方法，这个方法在同级目录的`handler.py`中，实际上还是指向了刚刚分析的对话函数`get_knowledge_based_answer()`

一些小知识：

- 在Python中，`__all__`是一个特殊的全局变量，它在模块中定义了一个列表，包含了你希望在`from module import *`语句中导出的所有名称。
如果一个模块定义了`__all__`，那么`from module import *`语句只会导入这个列表中的名称。

#### RAG API
QAngthing原始的API没有直接提供调用RAG的接口。In other words，查询就必须伴随内部Chat大模型的调用输出对话，这样会增加时间。
要想新增一个接口，需要修改三个文件：

修改步骤：
```shell
sudo docker exec -it qanything-container-local bash
cd qanything_local/qanything_kernel/
vim core/local_doc_qa.py
vim qanything_server/handler.py
vim qanything_server/sanic_api.py
```
`qanything_kernel/qanything_server/sanic_api.py`的`line 80`:
```python
app.add_route(local_doc_chat, "/api/local_doc_qa/local_doc_chat", methods=['POST'])
app.add_route(local_doc_search, "/api/local_doc_qa/local_doc_search", methods=['POST']) # [!code ++]
app.add_route(list_kbs, "/api/local_doc_qa/list_knowledge_base", methods=['POST'])
```

`qanything_kernel/qanything_server/handler.py`的`line 17`
```python
__all__ = ["new_knowledge_base", "upload_files", "list_kbs", "list_docs", "delete_knowledge_base", "delete_docs",
           "rename_knowledge_base", "get_total_status", "clean_files_by_status", "upload_weblink", "local_doc_chat",
           "document"] # [!code --]
           "local_doc_search", "document"] # [!code ++]
```
`line 423`新增一个函数
```python
async def local_doc_search(req: request):
    local_doc_qa: LocalDocQA = req.app.ctx.local_doc_qa
    user_id = safe_get(req, 'user_id')
    if user_id is None:
        return sanic_json({"code": 2002, "msg": f'�~S�~E��~]~^�~U�~Arequest.json�~Z{req.json}�~L请�~@�~_��~A'})
    is_valid = validate_user_id(user_id)
    if not is_valid:
        return sanic_json({"code": 2005, "msg": get_invalid_user_id_msg(user_id=user_id)})
    debug_logger.info('local_doc_chat %s', user_id)
    kb_ids = safe_get(req, 'kb_ids')
    question = safe_get(req, 'question')
    rerank = safe_get(req, 'rerank', default=True)
    debug_logger.info('rerank %s', rerank)
    streaming = safe_get(req, 'streaming', False)
    history = safe_get(req, 'history', [])
    debug_logger.info("history: %s ", history)
    debug_logger.info("question: %s", question)
    debug_logger.info("kb_ids: %s", kb_ids)
    debug_logger.info("user_id: %s", user_id)

    not_exist_kb_ids = local_doc_qa.milvus_summary.check_kb_exist(user_id, kb_ids)
    if not_exist_kb_ids:
        return sanic_json({"code": 2003, "msg": "fail, knowledge Base {} not found".format(not_exist_kb_ids)})

    file_infos = []
    milvus_kb = local_doc_qa.match_milvus_kb(user_id, kb_ids)
    for kb_id in kb_ids:
        file_infos.extend(local_doc_qa.milvus_summary.get_files(user_id, kb_id))
    valid_files = [fi for fi in file_infos if fi[2] == 'green']
    if len(valid_files) == 0:
        return sanic_json({"code": 200, "msg": "�~S�~I~M�~_��~F�~S为空�~L请�~J�| �~V~G件�~H~V�~I�~E�~V~G件解�~^~P�~L�~U", "question":
question,
                           "response": "All knowledge bases {} are empty or haven't green file, please upload files".format(
                               kb_ids), "history": history, "source_documents": [{}]})
    else:
        debug_logger.info("streaming: %s", streaming)
        if streaming:
            debug_logger.info("start generate answer")

            async def generate_answer(response):
                debug_logger.info("start generate...")
                for resp, next_history in local_doc_qa.get_knowledge_based_answer(
                        query=question, milvus_kb=milvus_kb, chat_history=history, streaming=True, rerank=rerank,
                        search_mode=True
                ):
                    chunk_data = resp["result"]
                    if not chunk_data:
                        continue
                    chunk_str = chunk_data[6:]
                    if chunk_str.startswith("[DONE]"):
                        source_documents = []
                        for inum, doc in enumerate(resp["source_documents"]):
                            source_info = {'file_id': doc.metadata['file_id'],
                                           'file_name': doc.metadata['file_name'],
                                           'content': doc.page_content,
                                           'retrieval_query': doc.metadata['retrieval_query'],
                                           'score': str(doc.metadata['score'])}
                            source_documents.append(source_info)

                        retrieval_documents = format_source_documents(resp["retrieval_documents"])
                        source_documents = format_source_documents(resp["source_documents"])
                        chat_data = {'user_info': user_id, 'kb_ids': kb_ids, 'query': question, 'history': history,
                                     'prompt': resp['prompt'], 'result': next_history[-1][1],
                                     'retrieval_documents': retrieval_documents, 'source_documents': source_documents}
                        qa_logger.info("chat_data: %s", chat_data)
                        debug_logger.info("response: %s", chat_data['result'])
                        stream_res = {
                            "code": 200,
                            "msg": "success",
                            "question": question,
                            # "response":next_history[-1][1],
                            "response": "",
                            "history": next_history,
                            "source_documents": source_documents,
                        }
                    else:
                        chunk_js = json.loads(chunk_str)
                        delta_answer = chunk_js["answer"]
                        stream_res = {
                            "code": 200,
                            "msg": "success",
                            "question": "",
                            "response": delta_answer,
                            "history": [],
                            "source_documents": [],
                        }
                    await response.write(f"data: {json.dumps(stream_res, ensure_ascii=False)}\n\n")
                    if chunk_str.startswith("[DONE]"):
                        await response.eof()
                    await asyncio.sleep(0.001)

            response_stream = ResponseStream(generate_answer, content_type='text/event-stream')
            return response_stream

        else:
            for resp, history in local_doc_qa.get_knowledge_based_answer(
                    query=question, milvus_kb=milvus_kb, chat_history=history, streaming=False, rerank=rerank,
                    search_mode=True
            ):
                pass
            retrieval_documents = format_source_documents(resp["retrieval_documents"])
            source_documents = format_source_documents(resp["source_documents"])
            chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, 'history': history,
                         'retrieval_documents': retrieval_documents, 'source_documents': source_documents}
            qa_logger.info("chat_data: %s", chat_data)
            return sanic_json({"code": 200, "msg": "success chat", "question": question, "response": "Well Done!",
                               "history": history, "source_documents": source_documents})
```


`qanything_kernel/core/local_doc_qa.py`的`line 219`
```python
    def get_knowledge_based_answer(self, query, milvus_kb, chat_history=None, streaming: bool = STREAMING,
            rerank: bool = False): # [!code --]
            rerank: bool = False, search_mode: bool = False): # [!code ++]
        if chat_history is None:
            chat_history = []
        retrieval_queries = [query]

        source_documents = self.get_source_documents(retrieval_queries, milvus_kb)

        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=True)
        if rerank and len(retrieval_documents) > 1:
            debug_logger.info(f"use rerank, rerank docs num: {len(retrieval_documents)}")
            retrieval_documents = self.rerank_documents(query, retrieval_documents)

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=PROMPT_TEMPLATE)
        prompt = self.generate_prompt(query=query, # [!code --]
                                      source_docs=source_documents, # [!code --]
                                      prompt_template=PROMPT_TEMPLATE) # [!code --]
        t1 = time.time()
        for answer_result in self.llm.generatorAnswer(prompt=prompt, # [!code --]
                                                      history=chat_history, # [!code --]
                                                      streaming=streaming): # [!code --]
            resp = answer_result.llm_output["answer"] # [!code --]
            prompt = answer_result.prompt # [!code --]
            history = answer_result.history # [!code --]
 # [!code --]
            # logging.info(f"[debug] get_knowledge_based_answer history = {history}") # [!code --]
            history[-1][0] = query # [!code --]
            response = {"query": query, # [!code --]
                        "prompt": prompt, # [!code --]
                        "result": resp, # [!code --]
                        "retrieval_documents": retrieval_documents, # [!code --]
                        "source_documents": source_documents} # [!code --]
            yield response, history # [!code --]
        if search_mode: # [!code ++]
            for _ in [1]: # [!code ++]
                history = [[None]] # [!code ++]
                history[-1][0] = query # [!code ++]
                response = {"query": query, # [!code ++]
                            "retrieval_documents": retrieval_documents, # [!code ++]
                            "source_documents": source_documents} # [!code ++]
                yield response, history # [!code ++]
        else: # [!code ++]
            prompt = self.generate_prompt(query=query, # [!code ++]
                                          source_docs=source_documents, # [!code ++]
                                          prompt_template=PROMPT_TEMPLATE) # [!code ++]
            t1 = time.time() # [!code ++]
            for answer_result in self.llm.generatorAnswer(prompt=prompt, # [!code ++]
                                                          history=chat_history, # [!code ++]
                                                          streaming=streaming): # [!code ++]
                resp = answer_result.llm_output["answer"] # [!code ++]
                prompt = answer_result.prompt # [!code ++]
                history = answer_result.history # [!code ++]
 # [!code ++]
                # logging.info(f"[debug] get_knowledge_based_answer history = {history}") # [!code ++]
                history[-1][0] = query # [!code ++]
                response = {"query": query, # [!code ++]
                            "prompt": prompt, # [!code ++]
                            "result": resp, # [!code ++]
                            "retrieval_documents": retrieval_documents, # [!code ++]
                            "source_documents": source_documents} # [!code ++]
                yield response, history # [!code ++]

        t2 = time.time()
        debug_logger.info(f"LLM time: {t2 - t1}")
```

使用：
```python
def search_content(question, verbose=True):
    url = ip_addr + ':8777/api/local_doc_qa/local_doc_search'
    headers = {
        'content-type': 'application/json'
    }
    data = {
        "user_id": "zzp",
        "kb_ids": ["KBee49e51dccce44df8932b4407b7d3015"],
        "question": question,
    }
    try:
        start_time = time.time()
        response = requests.post(url=url, headers=headers, json=data, timeout=60)
        end_time = time.time()
        res = response.json()
        if verbose:
          #  print(res['retrieval_documents'])
            print(f"响应状态码: {response.status_code}, 响应时间: {end_time - start_time}秒")
            for doc in res['source_documents']:
                print("#"*50)
                print(doc['file_name'])
                if len(doc['content']) > 200:
                    print(doc['content'][:200] + '......')
                else:
                    print(doc['content'])
                print(doc['score'])
    except Exception as e:
        print(f"请求发送失败: {e}")
```

### OCR
在项目的**Acknowledgments**部分可知，他们主要使用了百度飞桨的OCR，这里为了省事儿想直接用这个OCR，恰好这个工程也提供了接口。

在`qanything_kernel/core/local_doc_qa.py`中，可以看到
```python
        self.ocr_url = 'http://0.0.0.0:8010/ocr'

    def get_ocr_result(self, image_data: dict):
        response = requests.post(self.ocr_url, json=image_data, timeout=60)
        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
        return response.json()['results']

```

这里`get_ocr_result`在后续代码中作为`ocr_engine`，在`qanything_kernel/utils/loader/image_loader.py`中，可以看到
```python
            img_np = cv2.imread(filepath)
            h, w, c = img_np.shape
            img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
            result = self.ocr_engine(img_data)
```


因此，可以用如下方法蹭OCR:
首先代码为：
```python
import cv2
import base64
import requests

ocr_url = 'http://0.0.0.0:8010/ocr'


def get_ocr_result(image_data):
    response = requests.post(ocr_url, json=image_data, timeout=60)
    response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常
    return response.json()['results']


file_path = "test_img.jpg"
img_np = cv2.imread(file_path)
h, w, c = img_np.shape
img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
result = get_ocr_result(img_data)
result = [line for line in result if line]
print(result)
```

但是直接使用代码，发现容器内部的ocr服务连不上，这是因为需要加端口映射，由于项目使用 Docker Compose 管理容器，这里修改文件`docker-compose-linux.yaml`
```yaml
  qanything_local:
    container_name: qanything-container-local
    image: freeren/qanything:v1.2.2
    # runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: "all"
              capabilities: ["gpu"]

    command: /bin/bash -c 'if [ "${LLM_API}" = "local" ]; then /workspace/qanything_local/scripts/run_for_local_option.sh -c $LLM_API -i $DEVICE_ID -b $RUNTIME_BACKEND -m $MODEL_NAME -t $CONV_TEMPLATE -p $TP -r $GPU_MEM_UTILI; else /workspace/qanything_local/scripts/run_for_cloud_option.sh -c $LLM_API -i $DEVICE_ID -b $RUNTIME_BACKEND; fi; while true; do sleep 5; done'

    privileged: true
    shm_size: '8gb'
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/assets/custom_models:/model_repos/CustomLLM
      - ${DOCKER_VOLUME_DIRECTORY:-.}/:/workspace/qanything_local/
    ports:
      - "8777:8777"
      - "8010:8010" # [!code ++]
    environment:
      - NCCL_LAUNCH_MODE=PARALLEL
```

### 数据库
Milvus 是一个高度灵活、可靠且速度极快的云原生开源向量数据库。它为embedding 相似性搜索和AI 应用程序提供支持，并努力使每个组织都可以访问向量数据库。 Milvus 可以存储、索引和管理由深度神经网络和其他机器学习（ML）模型生成的十亿级别以上的embedding 向量。

在`qanything_kernel/connector/database/milvus/milvus_client.py`中可以看到，使用的库是[pymilvus](https://github.com/milvus-io/pymilvus/tree/master)

数据库保存在容器的`QANY_DB`文件夹中

## 其他
百川类似的模型
```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BaichuanTextEmbeddings
from dotenv import dotenv_values
env_vars = dotenv_values('.env')
# 百川向量算法
embedding = BaichuanTextEmbeddings(baichuan_api_key=env_vars['BAICHUAN_API_KEY'])
# 连接向量数据库
httpClient = chromadb.PersistentClient(path="./chromac")
# 文本分割
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
# chunk_size=400表示每次读取或写入数据时，数据的大小为400个字节, 约300~400个汉字
# 存入文档到数据库
def saveToVectorDB(file, collection):
    try:
        # 加载文本
        loader = TextLoader(file, encoding='gbk')  #中文必须带 encoding='gbk'
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        # 数据库实例化，并存入文档
        vectordb = Chroma(collection_name=collection, embedding_function=embedding, client=httpClient)
        ids = vectordb.add_documents_x(documents=docs)
        return "ok", 200
    except Exception as e:
        print(444, "saveDb error", e)
        return "error", 500
# 查询相似度的文本，同时根据分值过滤相关性
def queryVectorDB(ask, collection):
    vectordb = Chroma(collection_name=collection, embedding_function=embedding, client=httpClient)
    s = vectordb.similarity_search_with_score(query=ask, k=1)
    if len(s) == 0:
        return ""
    else:
        if s[0][1] < 1.35:   # 文本关联强则返回，不相关则不返回. openai < 0.385  baichuan < 1.35
            return s[0][0].page_content
        else:
            return ""
# 查询相似度的文本，返回所有数据
def queryText(ask, collection):
    vectordb = Chroma(collection_name=collection, embedding_function=embedding, client=httpClient)
    res = vectordb.similarity_search_all(query=ask, k=1)
    return res
# 删除文本
def deleteText(ids, collection):
    vectordb = Chroma(collection_name=collection, embedding_function=embedding, client=httpClient)
    res = vectordb.delete(ids=[ids])
    return "ok", 200
# 添加文本到数据库，返回结果id
def addText(text, collection):
    vectordb = Chroma(collection_name=collection, embedding_function=embedding, client=httpClient)
    res = vectordb.add_texts(texts=[text])
    return res
```

https://baoyu.io/translations/rag/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation

对话

1. 给每一个新对话计算embedding
2. 从头到尾将连续embedding差距小的聚集在一起，每一个聚类不超过10条
3. 给每一块计算embedding存入到数据库
4. 新对话计算embedding，匹配原始数据
5. 寻找最相近的几个

知识

1. 给每一句话计算embedding
2. 将embedding差距小的聚集在一起
3. 给每一块计算embedding存入到数据库
4. 新对话计算embedding，匹配原始数据
5. 寻找最相近的几个

embedding存储为json时提示float32报错：
`self.embeddings[k] = list(self.embeddings[k].astype('float'))`




[有道速读](https://read.youdao.com/#/home)

