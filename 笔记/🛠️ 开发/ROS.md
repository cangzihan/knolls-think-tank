---
tags:
  - 语音
  - TTS
---

# ROS

## 查看安装信息
```shell
printenv | grep ROS
```
如果看到类似于以下的输出，说明ROS已经安装并配置：
```shell
ROS_ROOT=/opt/ros/noetic/share/ros
ROS_PACKAGE_PATH=/opt/ros/noetic/share
ROS_MASTER_URI=http://localhost:11311
ROS_VERSION=1
ROS_PYTHON_VERSION=3
ROS_DISTRO=noetic
```

```shell
ROS_VERSION=2
ROS_PYTHON_VERSION=3
ROS_DOMAIN_ID=52
ROS_LOCALHOST_ONLY=0
ROS_DISTRO=humble
```

## 查看相机话题
```shell
ros2 topic list
```

## Python
默认是有关Python3的内容


```shell
conda install -c conda-forge libstdcxx-ng=12.2.0

strings $(conda info --base)/envs/py310_cv/lib/libstdc++.so.6 | grep GLIBCXX
ln -sf /path/to/required/libstdc++.so.6 /home/botler/anaconda3/envs/py310_cv/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python3 ros_image_test.py

```

### 列出所有节点
```python
import rclpy
from rclpy.node import Node

class NodeLister(Node):
    def __init__(self):
        super().__init__('node_lister')

    def list_nodes(self):
        node_names_and_namespaces = self.get_node_names_and_namespaces()
        node_names = [name for name, _ in node_names_and_namespaces]

        self.get_logger().info(f"Active ROS nodes:")
        for name, _ in node_names_and_namespaces:
            self.get_logger().info(f"\t{name}")



def main():
    rclpy.init()
    node_lister = NodeLister()

    # Ensure that the node is added to the global context before calling list_nodes
    rclpy.spin_once(node_lister, timeout_sec=0)
    node_lister.list_nodes()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Python
### rclpy
在ROS 2（Robot Operating System 2）中，`rclpy`是Python客户端库，用于与ROS 2的核心通信层（即中间件）交互。
它提供了用于Python的API，使得开发者可以在Python中轻松创建和管理节点、主题（topics）、服务（services）、**动作（actions）**等ROS 2的基本组件。

主要功能包括：
1. 节点创建：`rclpy`允许创建ROS 2节点，节点是ROS 2的基本计算单元。
2. 主题通信：支持发布和订阅主题，用于节点间的无状态数据流通信。
3. 服务调用：提供调用服务和定义服务的接口，支持节点间的请求/响应模式。
4. 动作接口：支持动作服务，适合处理需要时间的操作，例如机器人运动控制。
5. 参数管理：支持在运行时设置和获取节点参数，用于动态配置。
6. 计时器和执行器：rclpy提供定时器和事件循环，帮助在指定时间内定期执行任务。

`rclpy`是ROS 2的跨语言支持层（RCL）在Python中的实现，它使得Python可以直接访问ROS 2的C++核心功能，从而实现高效的机器人应用开发。
