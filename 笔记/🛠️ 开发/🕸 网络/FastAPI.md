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

#### 示例
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

特点
| 功能          | 说明                       |
| ----------- | ------------------------ |
| ORM 支持      | 像 SQLAlchemy 一样操作数据库     |
| Pydantic 兼容 | 自动数据验证                   |
| 类型注解友好      | 完全使用 Python typing       |
| 异步支持        | 可与 `async SQLAlchemy` 结合 |

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
