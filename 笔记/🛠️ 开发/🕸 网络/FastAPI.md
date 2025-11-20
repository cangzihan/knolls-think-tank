---
tags:
  - 后端
  - API
  - 通信
---

# FastAPI
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
- Documentation: https://fastapi.tiangolo.com
- 中文教程: https://fastapi.tiangolo.com/zh/tutorial/
- Source Code: https://github.com/fastapi/fastapi
- 官方模板: https://github.com/fastapi/full-stack-fastapi-template
- FastAPI Radar(一个三方可视化面板): https://github.com/doganarif/fastapi-radar

## Install
```shell
pip install "fastapi[standard]"
```

## 命令
### 启动服务
从 FastAPI 0.111.0（2024年5月左右） 开始，官方引入了一个新命令：
```shell
fastapi dev main.py
```

1. `fastapi dev`
    - 内置开发服务器（底层还是用 uvicorn + watchfiles 热重载）。
    - 适合本地调试。
    - 不需要单独安装 uvicorn，因为 FastAPI 依赖里已经帮你装好了。
    - 功能类似于 Flask 的 flask run。

2. `uvicorn main:app --reload`
    - 手动指定用 uvicorn 运行。
    - 更灵活，可以加参数（如 --workers 4）。
    - 适合开发和生产。

## 基本界面
Interactive API docs: http://127.0.0.1:8000/docs

Alternative API docs: http://127.0.0.1:8000/redoc

## 组件
### APIRouter
`APIRouter`是一个用于组织和模块化路由（即 API 路径操作）的核心组件。它允许你将相关的路由逻辑分组到不同的模块或文件中，然后再将这些模块集成到主 FastAPI 应用中，从而提高代码的可维护性和可读性。
```python
# users.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/")
def read_users():
    return [{"name": "Alice"}, {"name": "Bob"}]

@router.get("/{user_id}")
def read_user(user_id: int):
    return {"user_id": user_id, "name": "Alice"}

```
```python
# main.py
from fastapi import FastAPI
from users import router as user_router

app = FastAPI()

app.include_router(user_router)

```

### BackgroundTasks
BackgroundTasks 是 FastAPI 提供的一个工具，用于在 HTTP 响应发送给客户端后，在后台执行一些耗时的任务。这样可以快速响应客户端，同时在后台处理不需要立即返回给用户的操作。

```python
from fastapi import FastAPI, BackgroundTasks
import time

app = FastAPI()

# 模拟耗时操作的函数
def send_email(email: str, message: str):
    """模拟发送邮件"""
    print(f"Sending email to {email}: {message}")
    time.sleep(2)  # 模拟发送邮件的耗时
    print(f"Email sent to {email}")

def update_user_stats(user_id: str):
    """模拟更新用户统计"""
    print(f"Updating stats for user {user_id}")
    time.sleep(1)  # 模拟更新操作
    print(f"Stats updated for {user_id}")

def log_activity(user_id: str, action: str):
    """模拟记录活动"""
    print(f"Logging activity: {user_id} - {action}")
    time.sleep(0.5)  # 模拟记录操作
    print(f"Activity logged: {action}")

@app.get("/fast-operation")
async def fast_operation(background_tasks: BackgroundTasks):
    # 立即返回响应
    response = {"message": "Operation started", "status": "success"}
    
    # 在后台执行耗时操作
    background_tasks.add_task(send_email, "user@example.com", "Welcome!")
    background_tasks.add_task(update_user_stats, "user123")
    background_tasks.add_task(log_activity, "user123", "visited_page")
    
    return response  # 用户立即收到响应，后台任务继续执行

```

### Depends
`fastapi.Depends` 是 FastAPI 中实现依赖注入（Dependency Injection）的核心机制。它用于声明一个“依赖项”（dependency），FastAPI 会在处理请求前自动解析并注入这个依赖，常用于：
- 用户认证（如获取当前用户）
- 数据库会话管理
- 权限校验
- 共享逻辑复用（避免重复代码）

### UploadFile
UploadFile 是 FastAPI 提供的专门用于处理文件上传的类。它提供了一个异步友好的接口来处理上传的文件，支持流式处理，避免将整个文件加载到内存中。
```python
from fastapi import FastAPI, File, UploadFile
import shutil

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": len(await file.read())  # 注意：读取后文件指针会移动到末尾
    }

```

### security
#### OAuth2PasswordBearer
`fastapi.security.OAuth2PasswordBearer`是 FastAPI 中用于处理 基于 OAuth2 密码模式（Password Flow） 的认证组件。它本身 不实现认证逻辑，而是提供一个依赖项（Dependency），用于：
- 从请求头中自动提取 Bearer Token（即 JWT 或其他格式的访问令牌）；
- 集成到 OpenAPI 文档（Swagger UI / ReDoc），让你可以直接在 UI 中输入 token 进行测试；
- 作为依赖项注入到路径操作函数中，便于后续自定义认证逻辑。

⚠️ 注意事项
- `OAuth2PasswordBearer` **仅负责提取 token**，**不负责验证用户名/密码或生成 token**。
- 生成 token 的逻辑需要你自己实现（通常在 `/token` 路径中）。

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 模拟验证用户
def fake_hash_password(password: str):
    return "fakehashed" + password

fake_users_db = {
    "alice": {"username": "alice", "hashed_password": "fakehashedsecret"}
}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not user["hashed_password"] == fake_hash_password(form_data.password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    return {"token": token}

```

## 测试
https://fastapi.tiangolo.com/zh/tutorial/testing/

### 安装pytest
```shell
pip install pytest
```

### 测试脚本编写
文件夹结构
```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   └── test_main.py
```

测试脚本
::: code-group
```python [main.py]
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

```
```python [test_main.py]
# 用于在不运行真实服务器的情况下模拟客户端请求（如 GET、POST）
from fastapi.testclient import TestClient

# 导入要测试的 FastAPI 应用实例（app）
from .main import app

# 创建一个测试客户端对象，用于模拟 HTTP 请求
client = TestClient(app)


# 定义一个测试函数，用 pytest 或 unittest 等框架运行
def test_read_main():
    # 使用客户端向根路径（"/"）发送 GET 请求
    response = client.get("/")

    # 断言（assert）响应状态码应为 200，表示请求成功
    assert response.status_code == 200
    
    # 断言返回的 JSON 数据应等于 {"msg": "Hello World"}
    assert response.json() == {"msg": "Hello World"}

```
:::

::: details 关于`assert`语句
`assert`是 Python 自带的断言语句，主要用于 测试和调试，用来验证某个条件是否为真。

如果条件为 假 (False)，程序会抛出 AssertionError 异常并终止执行。

基本语法：
```python
assert 条件表达式, "可选的错误提示"
```

举个简单例子：
```python
x = 5
assert x > 0       # ✅ 条件为真，程序继续执行
assert x < 0, "x必须是负数"  # ❌ 条件为假，程序会报错

```

运行结果：
```shell
AssertionError: x必须是负数
```


:::

### 执行测试
```shell
pytest
```

## Debug
https://fastapi.tiangolo.com/zh/tutorial/debugging/

直接导入`uvicorn`并运行，可以在PyCharm中直接运行（施加断点后，【右键】-【调试】）下面代码进行调试特定接口。

```python
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    a = "a"
    b = "b" + a
    return {"hello world": b}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

## 常用库
### Pydantic
#### 功能
Pydantic 是 FastAPI 的“数据模型层”。
FastAPI 里所有的输入（请求体、参数）和输出（响应）验证，都是通过 Pydantic 实现的。

Pydantic 的作用就是：

    “把外部输入的 JSON、Dict 等原始数据 → 转换成带类型的 Python 对象，并自动验证格式。”

#### Basemodel
```python
from pydantic import BaseModel

# 定义一个请求体模型
class User(BaseModel):
    name: str
    age: int
    email: str

# FastAPI 自动验证请求体
from fastapi import FastAPI

app = FastAPI()

@app.post("/users/")
def create_user(user: User):  # 自动验证 JSON
    return {"message": f"Hello {user.name}, age {user.age}"}

```

功能
- 自动验证类型（`int`, `str`, `float` 等）；
- 自动生成文档；
- 自动提示错误。

#### Field
Field() 用来在 模型字段上声明额外信息：
- 默认值
- 描述（description）
- 标题（title）
- 校验规则（gt/ge/lt/le/regex等）
- 元数据（json_schema_extra）
- 别名（alias）
- 是否必须字段
- 示例（examples）

相当于：给字段添加约束 + 文档信息 + 默认值等定义
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(..., description="用户名", min_length=1, max_length=10)
    age: int = Field(18, ge=0, le=120, description="年龄")

```
解释：
- `name` 是必填字段（...）
- `age` 默认值 18
- 限制：0 ≤ `age` ≤ 120
- 文档描述会被 FastAPI 读取展示

#### field_serializer
它用于序列化字段时进行处理，也就是 对象 → dict/json 时生效

⚠️ 注意：不是解析（validation）时，而是输出时！

常见用途：
- 输出时格式化日期
- 隐藏敏感信息
- 输出时把某字段变成字符串
- 输出时修改字段结构
- 输出时自动拼接 UR

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class User(BaseModel):
    name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, created_at, info):
        return created_at.strftime("%Y-%m-%d %H:%M:%S")

u = User(name="Tom", created_at=datetime.now())
print(u.model_dump())

```

输出
```text
{
    "name": "Tom",
    "created_at": "2025-11-17 22:15:00"
}
```

隐藏敏感数据
```python
class User(BaseModel):
    username: str
    password: str

    @field_serializer("password")
    def hide_password(self, value, info):
        return "********"

```

#### model_validator
它是 Pydantic v2 中用于对 整个模型级别 进行校验的装饰器。区别于`field_validator`校验某个字段。

```python
class ImageRequest(BaseModel):
    prompt: str

    @model_validator(mode="before")
    def add_suffix(cls, data):
        data["prompt"] += ", Chinese"
        return data

```

```python
from datetime import datetime
from pydantic import BaseModel, model_validator, ValidationError

class Person(BaseModel):
    name: str
    age: int
    birth_year: int

    @model_validator(mode="after")
    def check_logic(self):
        current_year = datetime.now().year
        if current_year - self.birth_year != self.age:
            raise ValueError("年龄和出生年份不匹配")
        return self
```

### SQLModel
SQLModel 是由 FastAPI 作者 Sebastián Ramírez 自己开发的库，
可以理解为：

“把 SQLAlchemy（ORM） + Pydantic（数据验证） 融合成一个更易用的模型层。”

它继承了 SQLAlchemy 的所有特性，并且自动兼容 Pydantic 模型类型验证。
非常适合 FastAPI 一体化开发。

#### Install
```shell
pip install sqlmodel
```

#### 示例
```python
from sqlmodel import SQLModel, Field, create_engine, Session

# 定义数据表模型（自动继承 Pydantic 功能）
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int

# 创建数据库引擎
engine = create_engine("sqlite:///database.db")

# 创建表
SQLModel.metadata.create_all(engine)

# 插入数据
with Session(engine) as session:
    user = User(name="Yuchen", age=26)
    session.add(user)
    session.commit()

# 查询
with Session(engine) as session:
    users = session.query(User).all()
    print(users)

```

读数据
```python
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional

# 1. 定义模型
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: Optional[str] = None

# 2. 连接数据库（替换成你的密码）
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/mydb"
engine = create_engine(DATABASE_URL, echo=False)

# 3. 查询数据
def main():
    ascending=True
    with Session(engine) as session:
        # 构建查询语句
        statement = select(Event)
        
        if ascending:
            statement = statement.order_by(Event.time)      # 升序：最早 → 最晚
        else:
            statement = statement.order_by(Event.time.desc())  # 降序：最晚 → 最早
        
        events = session.exec(statement).all()
        for u in events:
            print(f"ID: {u.id}, Name: {u.name}, Email: {u.email}")

if __name__ == "__main__":
    main()

```

特点
| 功能          | 说明                       |
| ----------- | ------------------------ |
| ORM 支持      | 像 SQLAlchemy 一样操作数据库     |
| Pydantic 兼容 | 自动数据验证                   |
| 类型注解友好      | 完全使用 Python typing       |
| 异步支持        | 可与 `async SQLAlchemy` 结合 |


#### Relationship
`sqlmodel.Relationship`是 SQLModel 中用于定义 模型之间关联关系（如一对多、多对一、一对一） 的核心组件。它底层基于 SQLAlchemy 的 relationship()，但做了 Pydantic/SQLModel 友好的封装，让你能像处理普通属性一样操作关联对象。

基本语法
```python
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class Team(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

    # 一对多：一个 Team 有多个 Hero
    heroes: List["Hero"] = Relationship(back_populates="team")

class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    team_id: Optional[int] = Field(default=None, foreign_key="team.id")  # ← 外键

    # 多对一：一个 Hero 属于一个 Team
    team: Optional[Team] = Relationship(back_populates="heroes")
```

---

1. `back_populates`
    - **必须成对出现**：两边都要写，且值互指对方的字段名
    - 作用：告诉 SQLModel 如何**双向同步关系**
        - 当你设置 `hero.team = some_team`，`some_team.heroes` 会自动包含这个 hero
        - 当你往 `team.heroes.append(hero)`，`hero.team` 会自动指向该 team

2. 外键字段 (`foreign_key`)
    - `team_id: Optional[int] = Field(foreign_key="team.id")`
    - 这才是**真正创建数据库外键约束**的地方
    - `Relationship` 本身不创建外键，只提供对象访问层

3. 类型提示
    - 一对多：`List["Hero"]`（注意用字符串避免循环导入）
    - 多对一：`Optional[Team]`
    - 一对一：`Optional[Profile]`（两边都用 `Optional`）

### sse-starlette
sse-starlette 是一个轻量级的库，用于在 Starlette（以及基于 Starlette 的框架，如 FastAPI）中实现 SSE（Server-Sent Events，服务器发送事件） 功能。

#### 什么是 SSE（Server-Sent Events）？
SSE 是一种 服务器向客户端推送实时数据 的 Web 技术，基于 HTTP 长连接，具有以下特点：

| 特性 | 说明 |
|------|------|
| **单向通信** | 服务器 → 客户端（客户端不能通过 SSE 发消息给服务器） |
| **基于 HTTP/HTTPS** | 不需要 WebSocket 那样的特殊协议 |
| **自动重连** | 浏览器断开后会自动尝试重连 |
| **文本数据** | 只支持字符串（但可传 JSON） |
| **简单轻量** | 比 WebSocket 更容易实现和调试 |
#### Install
```shell
pip install sse-starlette
```

常见使用场景：
1. **AI 聊天机器人流式输出**（如 LLM 逐 token 返回）
2. **实时日志查看**
3. **股票价格/传感器数据更新**
4. **进度通知**（如文件处理进度）

#### FastAPI流式接口实现
```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境可设为 "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

TENG_WANG_GE_SHI = [
    "滕王高阁临江渚，佩玉鸣鸾罢歌舞。",
    "画栋朝飞南浦云，珠帘暮卷西山雨。",
    "闲云潭影日悠悠，物换星移几度秋。",
    "阁中帝子今何在？槛外长江空自流。"
]

@app.get("/poem/stream")
async def stream_poem():
    async def poem_generator():
        for line in TENG_WANG_GE_SHI[:-1]:
            yield line
            await asyncio.sleep(2)
        yield TENG_WANG_GE_SHI[-1]
        yield "[DONE]"
    return EventSourceResponse(poem_generator())

# 新增：返回前端页面
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
```

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>流式输出</title>
    <style>
        body {
            font-family: "Microsoft YaHei", sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        button {
            padding: 12px 24px;
            font-size: 18px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        button:disabled {
            background-color: #bdc3c7;
            cursor: not-allowed;
        }
        #poem-output {
            width: 100%;
            min-height: 200px;
            padding: 16px;
            font-size: 18px;
            line-height: 1.6;
            border: 1px solid #ccc;
            border-radius: 6px;
            background-color: white;
            white-space: pre-wrap; /* 保留换行和空格 */
            overflow-wrap: break-word;
        }
    </style>
</head>
    <body>
        <h1> 流式吟诵</h1>
        <div class="container">
            <button id="start-btn">开始接收</button>
            <div id="poem-output"></div>
        </div>
        <script>
            const startBtn = document.getElementById('start-btn');
            const output = document.getElementById('poem-output');

            let eventSource = null;
            let linesQueue = [];      // 待显示的句子队列
            let isTyping = false;     // 是否正在打字
            let currentLine = '';     // 当前正在打的句子
            let charIndex = 0;        // 当前句子已打到第几个字符

            // 打字机：逐字显示当前行
            function typeNextChar() {
                if (charIndex < currentLine.length) {
                output.textContent += currentLine[charIndex];
                charIndex++;
                setTimeout(typeNextChar, 100); // 每100ms打一个字
                } else {
                // 本行打完，换行
                output.textContent += '\n';
                isTyping = false;
                processQueue(); // 尝试处理下一行
                }
            }

            // 处理队列中的下一行
            function processQueue() {
                if (isTyping || linesQueue.length === 0) return;
                
                currentLine = linesQueue.shift();
                charIndex = 0;
                isTyping = true;
                typeNextChar();
            }

            startBtn.addEventListener('click', () => {
                if (eventSource) return;

                // 重置状态
                linesQueue = [];
                isTyping = false;
                output.textContent = '';
                startBtn.disabled = true;
                startBtn.textContent = '吟诵中...';

                eventSource = new EventSource('http://localhost:8000/poem/stream');

                eventSource.onmessage = (event) => {
                if (event.data === "[DONE]") {
                    eventSource.close();
                    eventSource = null;
                    startBtn.disabled = false;
                    startBtn.textContent = '重新开始';
                    return;
                }

                // 收到新句子，加入队列
                linesQueue.push(event.data);
                processQueue(); // 尝试开始打字
                };

                eventSource.onerror = () => {
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
                startBtn.disabled = false;
                startBtn.textContent = '重新开始';
                };
            });
        </script>
    </body>
</html>

```

### typing
#### Annotated
`typing.Annotated`是一个元数据标注工具，它允许你在类型注解中附加额外的“注释”（metadata），而不改变类型本身。
```python
from typing import Annotated
from fastapi import FastAPI, Depends, Query

app = FastAPI()

def get_user(token: str) -> dict:
    return {"token": token}

@app.get("/items/")
def read_items(
    user: Annotated[dict, Depends(get_user)],
    q: Annotated[str | None, Query()] = None,
):
    return {"user": user, "q": q}

```

### Uvicorn
FastAPI 不是一个“独立运行的服务器”，它只是一个 ASGI 应用。
而 Uvicorn 就是启动 FastAPI 应用的 ASGI Server（运行引擎）。

    就像 Flask 用 Werkzeug 启动一样，FastAPI 用 Uvicorn 启动。

```shell
uvicorn main:app --reload
```

解释：
- main → Python 文件名（main.py）
- app → FastAPI 实例名
- --reload → 热重载（开发时自动重启）
