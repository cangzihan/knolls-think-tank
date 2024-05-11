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
sudo bash ./run.sh -c local -i 4 -b default
```

- `-i`控制device_id
- `-c`设定`llm_api`
- `-b`为`default`时，且`llm_api`不为`cloud`时，使用7B模型。（`-b`默认为`default`）

网页端：http://[ip地址]:8777/qanything/#/home

[接口文档](https://github.com/netease-youdao/QAnything/blob/master/docs/API.md)

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

