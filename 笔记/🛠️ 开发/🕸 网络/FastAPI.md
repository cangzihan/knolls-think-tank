
# FastAPI
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
- Documentation: https://fastapi.tiangolo.com
- Source Code: https://github.com/fastapi/fastapi
- 官方模板: https://github.com/fastapi/full-stack-fastapi-template
- FastAPI Radar(一个三方可视化面板): https://github.com/doganarif/fastapi-radar

## Install
```
pip install "fastapi[standard]"
```

## 命令
### 启动服务
从 FastAPI 0.111.0（2024年5月左右） 开始，官方引入了一个新命令：
```
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
