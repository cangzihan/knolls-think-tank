---
tags:
  - Windows
---

# Windows
## Dell经常蓝屏

原因：https://learn.microsoft.com/zh-cn/troubleshoot/windows-server/performance/stop-code-driver-verifier-dma-violation

解决：
DMA蓝屏报错 可以试一下以下的处理方法：关机状态下按一下开机键然后马上反复按F2，进入BIOS界面进入之后选择Virtualization Support找到Enable Pre-Boot DMA Support，关闭这个选项然后找到Enable OS Kernel DMA Support，关闭这个选项然后点击APPLYCHANGES，保存然后点击EXIT退出，完成

作者：Unique
链接：https://www.zhihu.com/question/532059185/answer/2564226046
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
