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

## LangGraph
LangGraph 是一个用于构建状态化、多步骤 AI 应用的框架，特别适合需要对话、记忆、复杂工作流的应用。

Doc: [EN](https://docs.langchain.com/oss/python/langgraph/quickstart) | [中文](https://langgraph.com.cn/index.html)


### Install
```shell
pip install -U langgraph
```

### Graph
Graph 是 LangGraph 的核心概念，用来定义 AI 应用的工作流程。

LangGraph 的核心是将代理工作流建模为图。您可以使用三个关键组件来定义代理的行为：
1. State：A shared data structure that represents the current snapshot of your application. It can be any data type, 但通常是`TypedDict`或Pydantic `BaseModel`。
2. Nodes：Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state.
3. Edges：Python函数，根据当前的State决定接下来执行哪个Node。它们可以是条件分支或固定转换。

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

#### State
State可以理解为节点之间相互传递消息时，使用的通信格式。

Reducers
```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

```

#### Node
- `START`节点是一个特殊节点，代表将用户输入发送到图的节点。引用此节点的主要目的是确定应首先调用哪些节点。
- `END`节点是一个特殊节点，代表终止节点。当您想表示哪些边在完成操作后没有后续动作时，会引用此节点。

#### Edges
Edges define how the logic is routed and how the graph decides to stop. A node can have multiple outgoing edges. If a node has multiple outgoing edges, all of those destination nodes will be executed in parallel as a part of the next superstep.
```python
graph.add_edge("node_a", "node_b")

graph.add_conditional_edges("node_a", routing_function)
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})

```


### 自定义模型
```python
import operator
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

# ========== 1. 定义 State ==========
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ========== 2. 初始化自定义 LLM ==========
# 使用自定义模型（兼容 OpenAI API）
llm = ChatOpenAI(
    model="your-model-name", 
    base_url="http://localhost:11434/v1",  # 指向你的服务
    api_key="not-needed",         # 如果服务不需要 key，可填任意字符串
    temperature=0.7,
)

# ========== 3. 定义节点 ==========
def call_model(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ========== 4. 构建 Graph ==========
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

app = workflow.compile()

# ========== 5. 调用 ==========
if __name__ == "__main__":
    result = app.invoke({"messages": [HumanMessage(content="你好，你是谁？")]})
    print(result["messages"][-1].content)

```

### Tool
```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

# 1. 定义工具（推荐用 @tool 装饰器，确保有 schema）
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool(return_direct=True) # 立即返回工具结果并停止代理循环。
def number_to_hiragana(n: int) -> str:
    """Translate number to Japanese(Hiragana)."""
    if not (0 <= n <= 99999):
        raise ValueError("输入必须是 0 到 99999 之间的整数")
    
    if n == 0:
        return "ぜろ"

    # 数字到平假名的映射（1-9）
    digits = ["", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]
    
    man = n // 10000
    sen = (n % 10000) // 1000
    hyaku = (n % 1000) // 100
    juu = (n % 100) // 10
    ichi = n % 10

    parts = []

    # 处理“万”位
    if man > 0:
        parts.append(digits[man] + "まん")

    # 处理“千”位
    if sen > 0:
        if sen == 1 and man == 0:
            parts.append("せん")
        elif sen == 3:
            parts.append("さんぜん")
        elif sen == 8:
            parts.append("はっせん")
        else:
            parts.append(digits[sen] + "せん")

    # 处理“百”位
    if hyaku > 0:
        if hyaku == 1 and man == 0 and sen == 0:
            parts.append("ひゃく")
        elif hyaku == 3:
            parts.append("さんびゃく")
        elif hyaku == 6:
            parts.append("ろっぴゃく")
        elif hyaku == 8:
            parts.append("はっぴゃく")
        else:
            parts.append(digits[hyaku] + "ひゃく")

    # 处理“十”位
    if juu > 0:
        if juu == 1 and man == 0 and sen == 0 and hyaku == 0:
            parts.append("じゅう")
        else:
            parts.append(digits[juu] + "じゅう")

    # 处理个位
    if ichi > 0:
        # 特殊：4 和 7 在个位通常读「よん」「なな」，这里已包含在 digits 中
        parts.append(digits[ichi])

    return "".join(parts)

tools = [add, multiply, number_to_hiragana]

# 2. 使用OpenAI格式模型API
model = ChatOpenAI(
    model="llama3.2",               # 替换为你本地运行的模型名
    base_url="http://localhost:11434/v1",
    api_key="not-needed",
    temperature=0.7,
)

# 3. 绑定工具（禁用并行调用）
model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

# 4. 创建 agent
agent = create_react_agent(model=model_with_tools, tools=tools)

# 5. 调用
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what's 321 + 55 and 40 * 70? Your answer should be translated to Japanese hiragana."}]}
)

# 6. 打印结果
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        print(msg.content)

```

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
