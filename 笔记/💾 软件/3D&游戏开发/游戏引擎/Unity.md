---
tags:
  - 3D
  - 游戏开发
  - Unity
---

# Unity

Unity 界面
![img](assets/View.png)

## 视角

### 旋转
鼠标右键

### 平移
左右平移的方法： 鼠标右键 + A/D (同时按shift可以加速)
上下平移的方法： 鼠标右键 + Q/E (同时按shift可以加速)

方法2：
切换到Hand tool（快捷键Q）鼠标左键左右划

![img](assets/Toolbar.png)

### 放缩（前后）
- 方法1：鼠标滚轮
- 方法2：鼠标右键 + W/A (同时按shift可以加速)
- 方法3：Alt + 鼠标右键

## 对象

### 创建1个cube
right-click in the 【Hierarchy window】 and select 3D Object > Cube.

### 导入模型
直接把模型文件和材质贴图放到项目文件夹即可。
![img](assets/import_model.png)

## 图形化编程

If this were a C# script, you would have to play, stop, edit your script, and repeat until you found the values you wanted for your cube’s rotation. In C# scripting, you can’t edit your code while it runs, but in Visual Scripting, you can!

### 基本操作

#### 搜索（Fuzzy Finder）
鼠标右键

#### 删除线
右键点线的起点

## Project
### WFC 2019
项目地址：https://selfsame.itch.io/unitywfc

使用方法，打开任意1个Scene，Hierarchy找到map，然后在Inspector中点RUN，然后点generate

改变场景尺寸：修改Inspector中的Width和Depth
![img](assets/WFC.png)

按钮功能
在`SimpleTiledWFC.cs`中，通过TileSetEditor设置generate和RUN按钮

规则定义：
规则由map里的xml变量指向的文件控制，不是由canvas直接控制。但是可以修改万canvas后编译附带的【tiles】对象（或其他名字），然后点neigjbor那个按钮，就会自动修改xml。这样map就可以重新生成新规则的布局。

修改canvas:
选中canvas后，可以直接点上面的单元，然后在Inspector里有操作说明，比如按b换笔刷（模型），按x点击模型是清除模型。

让单元变大：改canvas和【tiles】（或其他名字）的grid_size，在此之前要清除所有的模型。生成时也要修改map的gridsize
