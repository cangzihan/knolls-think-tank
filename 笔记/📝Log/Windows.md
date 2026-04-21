---
tags:
  - Windows
---

# Windows
## 开机自启动任务

1. <kbd>Win</kbd> + <kbd>R</kbd>，输入`taskschd.msc` 
2. 右侧点击"Create Task..."（不是"Create Basic Task"）
3. General选项卡
  - Name填一个
  - 选择"Run whether user is logged on or not"
  - 选择"Run with highest privileges"
4. Triggers选项卡
  - New -> 把"On a schedule"改为"At startup"
5. Actions选项卡
  - New -> Program/script：选择写好的可执行文件
  - Start in：填写脚本所在的文件夹路径
6. Settings
  - 取消"Stop the task if it runs longer than"
  - 勾选"If the task fails, restart every"

启动：点击左边的"Task Scheduler Library"，然后在中间右键点击"Run"

## Dell经常蓝屏

原因：https://learn.microsoft.com/zh-cn/troubleshoot/windows-server/performance/stop-code-driver-verifier-dma-violation

解决：
DMA蓝屏报错 可以试一下以下的处理方法：关机状态下按一下开机键然后马上反复按F2，进入BIOS界面进入之后选择Virtualization Support找到Enable Pre-Boot DMA Support，关闭这个选项然后找到Enable OS Kernel DMA Support，关闭这个选项然后点击APPLYCHANGES，保存然后点击EXIT退出，完成

作者：Unique
链接：https://www.zhihu.com/question/532059185/answer/2564226046
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
