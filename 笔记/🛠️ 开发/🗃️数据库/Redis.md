# Redis
## Install
### Docker
```shell
# 拉取并运行 Redis 容器（前台运行也可，但建议后台）
docker run -d --name my-redis -p 6379:6379 redis:latest
```


## Python API
### Install
```shell
pip install redis
```

### 增/改
```python
# writer.py
import redis
import time

# 连接到 Docker 中的 Redis（localhost:6379）
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

key = "shared_message"

# 写入数据
message = "Hello from Writer! Current time: " + time.strftime("%Y-%m-%d %H:%M:%S")
r.set(key, message)

print(f"✅ 写入成功！Key: '{key}' -> Value: '{message}'")
```

过期时间：写入时可设置自动过期：
```python
r.set("temp_key", "value", ex=60)  # 60秒后自动删除
```

### 查
```
# reader.py
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

key = "shared_message"

# 读取数据
value = r.get(key)

if value is not None:
    print(f"✅ 读取成功！Key: '{key}' -> Value: '{value}'")
else:
    print(f"❌ Key '{key}' 不存在")
```
