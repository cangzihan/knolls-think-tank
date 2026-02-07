# 各种Python库

## logging
### 添加时间信息
```python
import logging

logger = logging.getLogger("reranker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"  # 可选：自定义时间格式
)
```

## requests
### SSL验证
跳过SSL验证
```python
import requests

response = requests.get('https://example.com', verify=False)

```

```python
import requests
from requests.exceptions import SSLError, RequestException

def safe_requests(method, url, **kwargs):
    """
    自动处理 SSL 证书错误的 requests 封装
    首次尝试 verify=True（默认），若 SSL 失败则重试 verify=False
    """
    try:
        # 第一次尝试：启用 SSL 验证（安全默认）
        return requests.request(method, url, **kwargs)
    except SSLError as e:
        print(f"SSL 验证失败，正在重试（verify=False）: {e}")
        # 第二次尝试：关闭 SSL 验证（⚠️ 不安全）
        kwargs.setdefault('verify', False)
        try:
            return requests.request(method, url, **kwargs)
        except Exception as retry_e:
            raise retry_e from e  # 保留原始异常链
    except RequestException:
        # 其他网络错误（如连接超时）直接抛出
        raise

# 使用示例
try:
    response = safe_requests('GET', 'https://example.com/api/data', timeout=10)
    response.raise_for_status()
    print(response.json())
except Exception as e:
    print(f"请求失败: {e}")

```

## aiohttp
[Project](https://docs.aiohttp.org/en/stable/)

aiohttp 是一个基于 asyncio 的异步 HTTP 客户端/服务器框架，专为 Python 的异步编程设计。

### 基本用法
#### HTTP 客户端示例
```python
import aiohttp
import asyncio

async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.github.com/users/octocat') as response:
            data = await response.json()
            print(data)

# 运行异步函数
asyncio.run(fetch_data())

```
#### Web 服务器示例
```python
from aiohttp import web

async def hello(request):
    return web.json_response({'message': 'Hello World!'})

app = web.Application()
app.router.add_get('/', hello)

if __name__ == '__main__':
    web.run_app(app, host='localhost', port=8080)
```

## celery
Celery 是一个分布式任务队列系统，用于处理大量消息，支持异步任务和定时任务执行。它主要用 Python 编写，但支持多种编程语言。

### 核心概念
架构组件
- Producer (生产者): 发送任务到队列
- Broker (中间件): 消息队列，如 Redis、RabbitMQ
- Worker (工作者): 执行任务的进程
- Backend (后端): 存储任务结果

工作流程

Producer → Broker → Worker → Backend

### Chain (链式任务)
`chain` 用于将多个任务按顺序链接执行，前一个任务的输出作为后一个任务的输入。
```python
from celery import chain, group
from celery_app import app

@app.task
def add(x, y):
    return x + y

@app.task
def multiply(x, factor):
    return x * factor

@app.task
def power(x, exp):
    return x ** exp

# 链式执行示例
def chain_example():
    # 执行顺序: add(2, 2) -> multiply(4, 3) -> power(12, 2)
    job = chain(
        add.s(2, 2),        # 返回 4
        multiply.s(3),      # 接收 4，返回 12
        power.s(2)          # 接收 12，返回 144
    )
    
    result = job.apply_async()
    print(result.get())  # 输出: 144

# 使用 .s() (signature) 语法
def chain_with_signature():
    # 等价于: power(multiply(add(2, 2), 3), 2)
    result = (add.s(2, 2) | multiply.s(3) | power.s(2)).apply_async()
    print(result.get())  # 输出: 144

```

### Group (并行任务)
```python
@app.task
def square(x):
    return x ** 2

@app.task
def double(x):
    return x * 2

def group_example():
    # 并行执行多个任务
    job = group(
        square.s(2),
        square.s(3),
        square.s(4),
        double.s(5)
    )
    
    result = job.apply_async()
    print(result.get())  # 输出: [4, 9, 16, 10]

# 与 chain 结合使用
def chain_group_example():
    # 先并行执行，然后将结果作为列表传递给下一个任务
    job = group(square.s(i) for i in range(3)) | sum_task.s()
    result = job.apply_async()
    print(result.get())  # 输出: 0² + 1² + 2² = 5

@app.task
def sum_task(numbers):
    return sum(numbers)

```

### 其他
`@shared_task` 是 Celery 提供的一个装饰器，它的主要作用是避免模块导入循环问题，让任务定义更加灵活。

传统方式（有潜在问题）：
```python
# celery_app.py
from celery import Celery

app = Celery('myproject')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

# tasks.py - 传统方式
from celery_app import app  # 必须导入 app

@app.task
def process_data(data):
    # 处理数据的逻辑
    return f"Processed: {data}"

@app.task
def send_email(recipient, subject, body):
    return f"Email sent to {recipient}"

# main.py
from tasks import process_data, send_email
from celery_app import app

# 可能出现导入循环问题
```

使用 @shared_task（推荐方式）：
```python
# celery_app.py (保持不变)
from celery import Celery

app = Celery('myproject')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

# tasks.py - 使用 @shared_task
from celery import shared_task

@shared_task
def process_data(data):
    """处理数据任务"""
    print(f"Processing: {data}")
    return f"Processed: {data}"

@shared_task
def send_email(recipient, subject, body):
    """发送邮件任务"""
    print(f"Sending email to {recipient}")
    return f"Email sent to {recipient}"

@shared_task
def calculate_sum(numbers):
    """计算数字总和"""
    total = sum(numbers)
    print(f"Sum of {numbers} = {total}")
    return total

# main.py - 使用任务
from tasks import process_data, send_email, calculate_sum

def main():
    # 调用任务
    result1 = process_data.delay("important data")
    result2 = send_email.delay("user@example.com", "Hello", "Welcome!")
    result3 = calculate_sum.delay([1, 2, 3, 4, 5])
    
    print("Task IDs:", result1.id, result2.id, result3.id)
    print("Results:", result1.get(), result2.get(), result3.get())

if __name__ == "__main__":
    main()

```

## multiprocessing
```python
from multiprocessing import Process
import time


def run(word):
    while True:
        print(word)
        time.sleep(2)

if __name__ == "__main__":
    # 创建第一个进程，目标函数是 run，参数是 ("XX",)
    # 注意：args=("XX",) 中的逗号很重要，这样 ("XX",) 才是一个元组，而不是字符串
    p1 = Process(target=run, args=("XX",))
    p1.start()    # 启动第一个进程
    
    # 创建第二个进程，目标函数是 run，参数是 ("OO",)
    p2 = Process(target=run, args=("OO",))
    p2.start()    # 启动第二个进程

    # 主进程的无限循环，防止主程序退出
    # 如果没有这个循环，主程序会立即结束，子进程也会被终止
    while True:
        # 主进程每秒检查一次，确保子进程继续运行
        time.sleep(1)

```

## torch
torch验证gpu是否可用并且如果可用列出全部GPU的信息
```python
import torch

# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    print("✅ CUDA 可用！")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print("-" * 50)
    
    # 获取 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个 GPU:")
    
    # 遍历每个 GPU 并打印详细信息
    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"  当前分配内存: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  已缓存内存: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("❌ CUDA 不可用。当前使用 CPU。")
```

## uv
uv是一个由 Astral（Ruff 团队）开发的超快 Python 包安装器和解析器。

`--system` 表示：将包安装到当前 Python 环境的全局 site-packages 目录中（即“系统 Python 环境”），而不是虚拟环境。

### 创建虚拟环境
```shell
uv venv

# 指定 Python 版本
uv venv --python 3.11
```

### 示例工作流
```shell
# 1. 创建项目目录
mkdir myproject && cd myproject

# 2. 创建虚拟环境
uv venv

# 3. 添加依赖
uv add requests httpx

# 4. 添加开发依赖
uv add --dev pytest ruff

# 5. 运行测试
uv run pytest

# 6. 格式化代码
uv run ruff format .

# 7. 导出 requirements.txt（如果需要）
uv pip compile pyproject.toml -o requirements.txt

```
