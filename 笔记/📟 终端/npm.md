
# npm

## 安装
### Linux
```shell
sudo nala install npm # or you can use apt
npm --version
```

## 启动本地服务器

```shell
npm run develop
npm run serve
HOST=0.0.0.0 npm run serve
npm run serve -- -l 3100
```

Error: Cannot find package 'typescript' imported from /mnt/data/Yuchen/super-splat/node_modules/@rollup/plugin-typescript/dist/es/index.js
```shell
npm install typescript
```

## 关闭
如果端口 3000 上的网页一直可以打开，即使你认为已经停止了相应的进程，可能有以下几种情况：

缓存或代理: 你的浏览器可能会缓存网页内容，或者代理服务器可能在后台继续提供相同的内容。尝试使用浏览器的无痕模式或清除缓存，或者通过其他网络访问网页。
重启问题: 如果你在更改配置后并没有重启你的服务器或服务进程，那么更改可能不会生效。确保在做出更改后重启相应的服务。
其他进程占用端口: 可能有其他进程正在监听端口 3000 并提供网页服务。你可以使用 netstat 命令检查端口的使用情况，并手动停止该进程。

```shell
netstat -tuln | grep 3000
```
然后根据结果，停止占用该端口的进程。
这里显示端口 3000 监听着一个 IPv6 地址，这意味着有一个进程在该端口上监听着 IPv6 连接。要停止这个进程，你可以使用 kill 命令或者其他适当的方法来结束这个进程。

首先，使用以下命令查找进程的 PID（进程 ID）：

```shell
sudo lsof -i :3000
```
然后，根据输出结果找到对应的进程 ID，并使用 kill 命令终止它：

```shell
sudo kill -9 <PID>
```
