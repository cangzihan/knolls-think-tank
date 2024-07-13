---
tags:
  - Flask
  - 通信
---

# Flask
Flask 是一个使用 Python 编程语言编写的轻量级 Web 应用框架。它由 Armin Ronacher 开发并于 2010 年首次发布。
Flask 的设计目标是简单易用且高度可扩展，适用于从小型项目到大型应用程序的广泛应用场景。


Flask 的应用场景
- 简单的 Web 应用：由于 Flask 轻量且易于使用，非常适合开发小型 Web 应用和微服务。
- API 服务：Flask 提供了简便的路由和 JSON 支持，适合用来开发 RESTful API。
- 快速原型开发：Flask 的简单性和灵活性使其成为快速原型开发的理想选择，开发者可以迅速构建和测试想法。
- 教学和学习：由于其简单的设计和易上手的特点，Flask 是学习 Web 开发基础概念和实践的良好起点。

## HTTP POST
**HTTP POST** 是 HTTP 协议的一种请求方法，主要用于向服务器提交数据。在 HTTP 请求方法中，POST 与 GET 最为常见。

**主要特点：**
1. 单向请求-响应模式：客户端发送请求到服务器，服务器处理请求后返回响应。
2. 有状态性：每次请求都是独立的，服务器不会保留任何状态信息（除非使用会话、cookies等）。
3. 安全性：POST 请求的数据包含在请求体中，相对于 GET 请求更不容易暴露在 URL 中。

**使用场景：**
- 表单提交。
- 上传文件。
- 调用 API 接口以进行数据处理。

### 代码
::: code-group
```python [客户端]
import requests

url = 'http://localhost:5000/api'
data = {'key': 'value'}

response = requests.post(url, json=data)
print(response.json())
```

```python [服务器(Flask)]
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api', methods=['POST']) # /api 是链接地址
def api():
    data = request.get_json()
    print(data)
    return jsonify({'response': 'Received data'})

if __name__ == '__main__':
    app.run(debug=True)
```
:::

## WebSocket
WebSocket 是一种全双工通信协议，允许客户端和服务器之间建立持久连接，并在该连接上进行双向数据传输。它是由 HTML5 提出的，并且与 HTTP 兼容。

主要特点：
1. 全双工通信：允许客户端和服务器同时发送和接收数据。
2. 低延迟：由于连接是持久的，不需要每次传输数据都重新建立连接，因此数据传输延迟较低。
3. 高效：与轮询等技术相比，WebSocket 可以显著减少网络开销，因为它在初始握手后不会有额外的 HTTP 请求和响应头。

使用场景：
- 实时应用程序（如在线聊天、游戏、股票交易等）。
- 需要低延迟、高频率数据更新的场景。
- 需要双向通信的场景。

**总结**
- WebSocket 适用于需要实时、低延迟的双向通信场景，例如在线聊天、实时更新的游戏或股票行情等。
- [HTTP POST](#http-post) 适用于一次性的数据提交和响应处理，例如表单提交、文件上传和调用 RESTful API 等。




### 代码
`flask_sock` 是一个用于 Flask 应用的 WebSocket 扩展。它简化了 WebSocket 的处理，让你可以在 Flask 应用中轻松添加 WebSocket 支持。

::: code-group

```python [服务器(Flask)]
from flask import Flask
from flask_sock import Sock, Server

app = Flask(__name__)
sock = Sock(app)

@sock.route('/echo')
def echo(ws: Server):
    while True:
        data = ws.receive()
        if data is None:
            break
        ws.send(data)

if __name__ == '__main__':
    app.run(debug=True)
```

```html [客户端 (JavaScript)]
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
</head>
<body>
    <h1>WebSocket Echo Test</h1>
    <input type="text" id="messageInput" placeholder="Type a message">
    <button onclick="sendMessage()">Send</button>
    <div id="output"></div>

    <script>
        var ws = new WebSocket('ws://localhost:5000/echo');

        ws.onopen = function(event) {
            console.log('WebSocket connection opened');
        };

        ws.onmessage = function(event) {
            var outputDiv = document.getElementById('output');
            var newMessage = document.createElement('div');
            newMessage.textContent = 'Received: ' + event.data;
            outputDiv.appendChild(newMessage);
        };

        ws.onclose = function(event) {
            console.log('WebSocket connection closed');
        };

        function sendMessage() {
            var input = document.getElementById('messageInput');
            var message = input.value;
            ws.send(message);
        }
    </script>
</body>
</html>
```
:::


JavaScript 不需要编译是因为它是解释型语言，浏览器可以直接理解和执行 JavaScript 代码。你只需将 JavaScript 代码嵌入到 HTML 文件中或通过 `<script>` 标签引入一个外部的 JavaScript 文件，浏览器就可以直接执行这些代码。

TypeScript 是 JavaScript 的超集，增加了类型系统等特性，需要通过编译器将 TypeScript 代码转换成 JavaScript 代码，浏览器才能执行。

Unity客户端：

1. 获取必要的`dll`文件：下载[WebSocketSharp页面](https://www.nuget.org/packages/WebSocketSharp.Standard)中的package，找到`websocket-sharp-standard.dll`。
将 `websocket-sharp-standard.dll` 复制到 Unity 项目的 `Assets/Plugins` 文件夹中。如果没有 `Plugins` 文件夹，请先创建一个。

2. 编写两个脚本

::: code-group

```cs [WebSocketClient.cs]
using UnityEngine;
using WebSocketSharp;
using System.Collections.Generic;
using TMPro;

public class WebSocketClient : MonoBehaviour
{
    private WebSocket ws;
    public TMP_InputField InputField;
    public TMP_Text OutputText;

    void Start()
    {
        // WebSocket 服务器地址
        string serverAddress = "ws://localhost:5000/echo";

        // 初始化 WebSocket
        ws = new WebSocket(serverAddress);

        // 设置消息接收事件处理函数
        ws.OnMessage += (sender, e) =>
        {
            // 确保在主线程中更新 UI
            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                Debug.Log("Received: " + e.Data);
                OutputText.text += "\n" + e.Data;
            });
        };

        // 连接到服务器
        ws.Connect();
    }

    void OnDestroy()
    {
        // 关闭 WebSocket 连接
        if (ws != null)
        {
            ws.Close();
        }
    }

    public void OnSendButtonClick()
    {
        // 发送消息到服务器
        if (ws != null && ws.IsAlive)
        {
            string message = InputField.text;
            ws.Send(message);
        }
    }
}

```

```cs [UnityMainThreadDispatcher.cs]
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UnityMainThreadDispatcher : MonoBehaviour
{
    private static readonly Queue<Action> _executionQueue = new Queue<Action>();

    private static UnityMainThreadDispatcher _instance = null;

    public static UnityMainThreadDispatcher Instance()
    {
        if (!_instance)
        {
            throw new Exception("UnityMainThreadDispatcher not initialized.");
        }
        return _instance;
    }

    void Awake()
    {
        if (_instance == null)
        {
            _instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    public void Enqueue(IEnumerator action)
    {
        lock (_executionQueue)
        {
            _executionQueue.Enqueue(() =>
            {
                StartCoroutine(action);
            });
        }
    }

    public void Enqueue(Action action)
    {
        Enqueue(ActionWrapper(action));
    }

    IEnumerator ActionWrapper(Action action)
    {
        action();
        yield return null;
    }

    void Update()
    {
        lock (_executionQueue)
        {
            while (_executionQueue.Count > 0)
            {
                _executionQueue.Dequeue().Invoke();
            }
        }
    }
}

```
:::

3. **Unity Main Thread Dispatcher**
由于 WebSocket 的回调可能在非主线程中执行，因此需要一个工具将回调任务派发到 Unity 主线程。你可以使用 UnityMainThreadDispatcher。以下是如何添加并使用它：
   1. 将`UnityMainThreadDispatcher.cs` 脚本添加到 Unity 项目中的 `Assets` 文件夹。

   2. 创建一个空的 GameObject，命名为 `MainThreadDispatcher`，然后将 `UnityMainThreadDispatcher` 脚本附加到该 GameObject。

4. 设置 Unity 场景

   1. 创建一个新的场景，添加一个 UI 输入框（TMP_InputField）、一个按钮和一个文本框（TMP_Text）。
   2. 将 `WebSocketClient` 脚本附加到一个空的 GameObject。
   3. 将输入框、按钮和文本框分别拖动到 `WebSocketClient` 脚本的 `InputField` 和 `OutputText` 引用字段中。
   4. 为按钮添加点击事件，将 `WebSocketClient` 的 `OnSendButtonClick` 方法绑定到按钮的点击事件中。

5. 运行项目

## 跨域请求
在 Flask 服务器中添加 CORS 头信息，允许来自不同来源的请求。

```shell
pip install flask-cors
```

```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
```


