---
tags:
  - 数据库
---

# Milvus

## 什么是 Milvus 向量数据库？
Milvus 是在 2019 年创建的，其唯一目标是存储、索引和管理由深度神经网络和其他机器学习（ML）模型生成的大规模嵌入向量。

作为一个专门设计用于处理输入向量查询的数据库，它能够处理万亿级别的向量索引。与现有的关系型数据库主要处理遵循预定义模式的结构化数据不同，Milvus 从底层设计用于处理从非结构化数据转换而来的嵌入向量。

随着互联网的发展和演变，非结构化数据变得越来越常见，包括电子邮件、论文、物联网传感器数据、Facebook 照片、蛋白质结构等等。为了使计算机能够理解和处理非结构化数据，使用嵌入技术将它们转换为向量。Milvus 存储和索引这些向量。Milvus 能够通过计算它们的相似距离来分析两个向量之间的相关性。如果两个嵌入向量非常相似，则意味着原始数据源也很相似。


## Install
通常一个含有Milvus的工程同时含有三种容器`milvus-etcd`, `milvus-minio`, `milvus-standalone`

https://milvus.io/docs/v2.1.x/install_standalone-docker.md
拉取Milvus镜像

```shell
docker pull milvusdb/milvus:v2.3.4
...
```

```shell
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
或者修改配置为：
```text
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd_1
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio_1
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone_1
    image: milvusdb/milvus:v2.3.4
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
```
~~在上面的配置中，我们将Milvus服务的端口映射到主机的19530和19121端口，并设置了环境变量TZ为Asia/Shanghai，以便使用上海时区。同时，我们将主机的./db_path目录映射到容器内的/var/lib/milvus/db目录，用于存储Milvus的数据。~~

In the same directory as the`docker-compose.yml`file, start up Milvus by running:
```shell
# 接下来，在终端中进入包含docker-compose.yml文件的目录，并运行以下命令启动Milvus服务：
sudo docker-compose up -d
```

Now check if the containers are up and running.
```shell
sudo docker-compose ps
```

### 基本命令
#### 连接数据库
```python
from pymilvus import connections

connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)
```

#### Collection
::: code-group
```python [添加]
from pymilvus import connections
connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)

# Prepare Schema
from pymilvus import CollectionSchema, FieldSchema, DataType
book_id = FieldSchema(
  name="book_id",
  dtype=DataType.INT64,
  is_primary=True,
)
book_name = FieldSchema(
  name="book_name",
  dtype=DataType.VARCHAR,
  max_length=200,
)
word_count = FieldSchema(
  name="word_count",
  dtype=DataType.INT64,
)
book_intro = FieldSchema(
  name="book_intro",
  dtype=DataType.FLOAT_VECTOR,
  dim=2
)
schema = CollectionSchema(
  fields=[book_id, book_name, word_count, book_intro],
  description="Test book search"
)

# Create a collection with the schema
collection_name = "book"
from pymilvus import Collection
collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2,
    )
```

```python [检查]
from pymilvus import connections
connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)

from pymilvus import utility
print("数据库是否有book的collection:", utility.has_collection("book"))
print("数据库是否有Facebook的collection:", utility.has_collection("Facebook"))
```

```python [删库]
from pymilvus import utility
utility.drop_collection("book")
```

:::

#### Partition
::: code-group
```python [添加]
from pymilvus import connections
connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)

from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.create_partition("novel")
```

```python [检查]
from pymilvus import connections
connections.connect(
  alias="default",
  host='localhost',
  port='19530'
)

from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
print(collection.partitions)
```

```python [删除]
from pymilvus import Collection
collection = Collection("book")      # Get an existing collection.
collection.drop_partition("novel")
```
:::

#### Data
```python
import random
data = [
  [i for i in range(2000)],
  [str(i) for i in range(2000)],
  [i for i in range(10000, 12000)],
  [[random.random() for _ in range(2)] for _ in range(2000)],
]
```

## 可视化工具
### 部署
#### Step 1: Install Attu
```shell
docker run -d --name attu -p 19600:3000 zilliz/attu:latest
```

The log shows the internal port of Attu, 3000.
```shell
docker logs attu
```

If not 3000, u should recreate docker(`docker rm`) with the correct port.

#### Step 2: Access Attu
Now, you should be able to access Attu by navigating to `http://SERVER_IP:19600` in your web browser, where SERVER_IP is the IP address of your server.

#### Step 3: Connect Attu to Milvus
When you first open Attu, you'll need to connect it to your Milvus instance:

Enter the **host** and **port** of your Milvus server (e.g., `localhost` and `19530`).
Click **Connect**.
