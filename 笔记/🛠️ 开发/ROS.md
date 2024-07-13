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
