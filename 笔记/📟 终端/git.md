---
tags:
  - Linux
---

# git

## Git配置Clash代理
```shell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy https://127.0.0.1:7890
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 同步克隆的仓库到最新
```shell
git pull origin main # master
```
