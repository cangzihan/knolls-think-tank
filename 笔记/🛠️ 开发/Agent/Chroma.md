---
tags:
  - 数据库
---

# Chroma
Chroma 是一个开源的向量数据库，专为 AI 应用程序设计，特别适用于检索增强生成 (RAG)、语义搜索和相似性搜索等场景。

[Doc](https://docs.trychroma.com/docs/overview/getting-started) | [Github](https://github.com/chroma-core/chroma) (其中有Colab笔记)

核心特性
1. 向量存储
    - 高效存储和检索向量嵌入
    - 支持高维向量数据
    - 优化的索引结构
2. 相似性搜索
    - 快速的最近邻搜索
    - 支持余弦相似度、欧几里得距离等多种距离度量
    - 实时相似性匹配
3. 元数据支持
    - 丰富的元数据存储
    - 支持过滤和分面搜索
    - 结构化数据关联

相关工具：
- https://github.com/msteele3/chromagraphic
- https://github.com/AYOUYI/chromadb_GUI_tool

## 数据库安装
事实上，使用Chromadb并不需要电脑真正安装数据库，当然安装数据库更好。ChromaDB 的一大优势就是：默认不需要你单独安装或运行一个数据库服务器（比如像 PostgreSQL、MongoDB 那样要先启动服务）。它开箱即用，特别适合快速开发、本地实验或嵌入式场景。

### Docker部署
使用`chromadb/chroma`镜像 https://hub.docker.com/r/chromadb/chroma

`docker-compose.yml`
```yml
version: '3.8'

services:
  chroma:
    image: chromadb/chroma:1.0.22.dev23
    container_name: chroma-db
    ports:
      - "8000:8000"
    environment:
      - IS_PERSISTENT=true
      - CHROMA_STORAGE_PATH=/chroma/chroma_data
    volumes:
      - ./chroma_data:/chroma/chroma_data
    restart: unless-stopped
```

运行`docker-compose up -d`

测试是否成功`curl http://localhost:8000/api/v1/heartbeat`

客户端
```python
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings()
)

collection = client.get_or_create_collection("test")
collection.add(documents=["Hello Chroma!"], ids=["1"])
print(collection.query(query_texts=["Hi"], n_results=1))

```

## Collection
在 Chroma（一个用于向量嵌入存储和检索的开源向量数据库）中，Collection（集合） 是一个核心概念。你可以把它理解为向量数据库中的“表”或“命名空间”，用来组织和管理一组相关的向量数据（embeddings）及其对应的元数据（metadata）和原始文档内容（documents）。

## Python API
### Install
你并不必要额外安装数据库服务器即可使用
```shell
pip install chromadb
```

### 启动服务

🔄 两种运行模式

| 模式 | 说明 | 是否需要安装服务 |
|------|------|----------------|
| **Local / Persistent Client（默认）** | 数据存在本地磁盘，单机使用 | ❌ 不需要 |
| **HTTP Client（远程服务器）** | 连接远程 Chroma 服务（如部署在 Docker、K8s） | ✅ 需要先部署 Chroma 服务端 |

- Chroma 默认是“无服务”的（serverless / embedded），用文件系统做存储，开箱即用。
- 你不需要安装数据库软件，pip install chromadb 就够了。
- 但你可以选择部署 Chroma 服务（比如用 Docker），用于生产或多用户场景。
- 本质上，Chroma 既是库（library），也能变成服务（service） —— 这是它的灵活之处。

#### 1. 本地模式
```python
import chromadb
client = chromadb.PersistentClient(path="./my_vector_db")  # 显式指定路径
# 或
client = chromadb.Client()  # 默认路径
```
#### 2. 远程模式
```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
```

### 检索
即使不提供 embedding 向量，Chroma 也能“自动”帮你做文本检索。Chroma 默认会自动为字符串计算 embedding！
它不是直接“按字符串匹配”检索，而是在你没提供 embedding 时，自动调用内置的嵌入函数（embedding function）把文本转成向量，再做语义搜索。

你没有传`embeddings`参数，Chroma 会：

1. 检查这个 collection 是否配置了`embedding_function`
2. 如果没有显式指定，Chroma 会使用默认的嵌入模型（目前是 `SentenceTransformer`的`all-MiniLM-L6-v2`）
3. 自动调用该模型，把`documents`中的每段文本转成向量（在 CPU 上计算！）
4. 存储向量 + 原始文本

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

client = chromadb.Client()

# 方式1：不指定 embedding_function（使用默认）
collection1 = client.create_collection("test1")

# 方式2：显式指定（和默认其实一样）
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection2 = client.create_collection("test2", embedding_function=ef)

# 添加文档（都不提供 embeddings）
collection1.add(documents=["猫在睡觉"], ids=["1"])
collection2.add(documents=["猫在睡觉"], ids=["1"])

```
这两个 collection 行为几乎一致——都在本地 CPU 上用 SentenceTransformer 算 embedding。

#### OpenAI embedding
```python
import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# 确保设置了 API key（也可以通过 openai.api_key = "..." 设置）
# os.environ["OPENAI_API_KEY"] = "sk-..."

# 创建 embedding function
openai_ef = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"  # 推荐：性价比高
    # model_name="text-embedding-ada-002"  # 旧版，也可用
)

# 创建 client 和 collection（自动使用 OpenAI 计算 embedding）
client = chromadb.Client()
collection = client.create_collection(
    name="my_openai_docs",
    embedding_function=openai_ef  # 关键：绑定 OpenAI 函数
)

# 添加文档（无需手动计算 embedding！）
collection.add(
    documents=[
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Chroma is an open-source vector database."
    ],
    metadatas=[
        {"category": "sentence"},
        {"category": "AI"},
        {"category": "database"}
    ],
    ids=["1", "2", "3"]
)

# 查询（自动用 OpenAI embedding query 文本）
results = collection.query(
    query_texts=["What is Chroma?"],
    n_results=2
)

print(results["documents"])

```

