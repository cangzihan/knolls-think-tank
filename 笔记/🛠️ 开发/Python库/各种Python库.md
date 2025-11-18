# 各种Python库

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

## LangGraph
LangGraph 是一个用于构建状态化、多步骤 AI 应用的框架，特别适合需要对话、记忆、复杂工作流的应用。

### Graph
Graph 是 LangGraph 的核心概念，用来定义 AI 应用的工作流程。

想象一下：你正在画一个流程图，描述任务的执行步骤。
```python
from langgraph.graph import StateGraph

# 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]
    current_task: str
    task_history: list

# 创建图
workflow = StateGraph(State)

# 添加节点（步骤）
def analyze_input(state):
    # 分析用户输入
    return {"current_task": "analyze"}

def process_data(state):
    # 处理数据
    return {"current_task": "process"}

def generate_response(state):
    # 生成回复
    return {"current_task": "respond"}

# 添加节点到图
workflow.add_node("analyze", analyze_input)
workflow.add_node("process", process_data)
workflow.add_node("respond", generate_response)

# 定义执行顺序
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "process")
workflow.add_edge("process", "respond")
workflow.add_edge("respond", "__end__")

# 编译图
app = workflow.compile()

```

Graph 的特点
- 状态管理：记住之前的对话和操作
- 条件分支：根据条件选择不同路径
- 循环执行：可以重复执行某些步骤

### Create React Agent（创建 ReAct 代理）
Prebuilt 是 LangGraph 提供的预构建好的图结构，可以直接使用，无需从零开始构建。

ReAct = Reasoning（推理）+ Act（行动）
- 推理：分析问题，思考解决方案
- 行动：执行工具，获取信息
- 重复：根据结果继续推理和行动

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网页获取信息"""
    # 模拟搜索结果
    return f"搜索结果：{query}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    # 安全的计算实现
    return str(eval(expression))

# 创建 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建 ReAct 代理
agent = create_react_agent(
    llm=llm,
    tools=[search_web, calculate]
)

# 使用代理
for chunk in agent.stream({
    "messages": [("user", "北京到上海的距离是多少公里？")]
}):
    print(chunk)

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

## uv
uv是一个由 Astral（Ruff 团队）开发的超快 Python 包安装器和解析器。

`--system` 表示：将包安装到当前 Python 环境的全局 site-packages 目录中（即“系统 Python 环境”），而不是虚拟环境。
