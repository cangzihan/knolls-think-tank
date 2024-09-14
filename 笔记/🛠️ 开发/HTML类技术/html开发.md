---
tags:
  - HTML
  - 网页
---

# html开发

HTML（HyperText Markup Language）即超文本标记语言，是用于创建网页的标准标记语言。它不是一种编程语言，而是一种用于定义网页内容结构和表现方式的标记语言。通过使用HTML，可以告诉浏览器如何显示网页上的文本、图片、链接、表格、列表等元素。HTML（HyperText Markup Language）即超文本标记语言，是用于创建网页的标准标记语言。它不是一种编程语言，而是一种用于定义网页内容结构和表现方式的标记语言。通过使用HTML，可以告诉浏览器如何显示网页上的文本、图片、链接、表格、列表等元素。

HTML 文档由一系列的HTML元素组成，这些元素通过标签（tags）来定义。标签通常成对出现，包括一个开始标签和一个结束标签，例如 `<p>` 和 `</p>` 用于定义一个段落（paragraph）。有些标签是自闭合的，比如 `<br />`，用于在HTML中插入一个换行符，它不需要结束标签。

HTML文档的基本结构如下：
```html
<!DOCTYPE html>
<html>
<head>
    <title>页面标题</title>
</head>
<body>

<h1>这是一个标题</h1>
<p>这是一个段落。</p>

</body>
</html>
```

- `<!DOCTYPE html>` 声明了文档类型和HTML版本。
- `<html>` 元素是根元素，所有的其他HTML元素都应该位于 `<html>` 元素内部。
- `<head>` 元素包含了文档的元（meta）数据，如 `<title>` 定义了文档的标题，这个标题会显示在浏览器的标题栏或页面的标签上。
- `<body>` 元素包含了可见的页面内容，如标题、段落、图片、链接、表格、列表等。

HTML5是HTML的最新版本，它引入了更多的语义化标签，比如 `<article>`、`<section>`、`<header>`、`<footer>`等，这些标签使得网页的结构更加清晰，也更有利于搜索引擎优化（SEO）和网页的可访问性。此外，HTML5还引入了一些新的API和元素，支持音频、视频、图形、动画等多媒体内容的嵌入，以及表单控件和拖放功能的增强。

通常情况下我不会编HTML，但是如果一个程序能完全不依赖Python，那么可以使用HTML做

## UI
### Button
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>What is ZS?（Button）</title>
<style>
  .btn-group button {
    margin-right: 10px;
  }
</style>
</head>
<body>

<h2>赵爽是个啥？</h2>

<div class="btn-group">
  <button type="button" class="option" onclick="selectOption(this)">傻逼</button>
  <button type="button" class="option" onclick="selectOption(this)">狗</button>
</div>

<p id="hint"></p> <!-- 预留位置，用于在这里显示提示 -->

<script>
  function selectOption(button) {

    // 显示提示信息（如果需要）
    var hint = document.getElementById("hint");
    hint.innerHTML = "赵爽是" + button.textContent + "！";
  }
</script>

</body>
</html>
```

### Radio
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>What is ZS?（Radio）</title>
<script>
  function showHint(option) {
    // 根据选项显示不同的提示
    var hint = document.getElementById("hint");
    if (option === "option1") {
      hint.innerHTML = "赵爽是傻逼！";
    } else if (option === "option2") {
      hint.innerHTML = "赵爽是狗！";
    } else {
      hint.innerHTML = ""; // 清除之前的提示
    }
  }
</script>
</head>
<body>

<h2>赵爽是个啥？</h2>

<p>
  <input type="radio" id="option1" name="color" value="option1" onclick="showHint(this.value)">
  <label for="option1">傻逼</label>
</p>

<p>
  <input type="radio" id="option2" name="color" value="option2" onclick="showHint(this.value)">
  <label for="option2">狗</label>
</p>

<p id="hint"></p> <!-- 预留位置，用于在这里显示提示 -->

</body>
</html>
```

### Input
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>回答问题</title>
</head>
<body>

<h2>请回答以下问题：</h2>
<p>赵爽是个啥？</p>
<input type="text" id="answerInput" placeholder="请输入答案">
<button onclick="submitAnswer()">提交</button>

<p>赵爽是<span id="answerDisplay"></span>！</p>

<script>
  // script.js 文件内容
    function submitAnswer() {
      // 获取输入框的值
      var answer = document.getElementById('answerInput').value;

      // 将答案显示在指定的元素中
      document.getElementById('answerDisplay').textContent = answer;
    }
</script>
</body>
</html>
```

### ScrollBar
在HTML叫做`select`
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>下拉列表示例</title>
</head>
<body>

<h2>赵爽是个啥？</h2>

<select id="colorSelector" onchange="showColorHint()">
  <option value="">请选择...</option>
  <option value="傻逼">傻逼</option>
  <option value="狗">狗</option>
</select>

<p id="hint"></p> <!-- 用于显示选中的提示 -->

<script>
  function showColorHint() {
    // 获取下拉列表的选中值
    var selectedColor = document.getElementById("colorSelector").value;

    // 显示提示信息
    var hint = document.getElementById("hint");
    if (selectedColor) { // 如果选中了某个颜色
      hint.innerHTML = "赵爽是" + selectedColor + "！";
    } else { // 如果没有选中任何（即选择了“请选择...”选项）
      hint.innerHTML = ""; // 清空提示信息
    }
  }
</script>

</body>
</html>
```

### CheckBox
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>多选框示例</title>
</head>
<body>

<h2>你最喜欢的颜色是什么？（可多选）</h2>

<form id="colorForm">
  <input type="checkbox" id="red" name="color" value="红色">
  <label for="red">红色</label><br>
  <input type="checkbox" id="blue" name="color" value="蓝色">
  <label for="blue">蓝色</label><br>
  <input type="checkbox" id="green" name="color" value="绿色">
  <label for="green">绿色</label><br>
</form>

<p id="hint"></p> <!-- 用于显示选中的提示 -->

<script>
  // 为每个多选框添加事件监听器
  document.querySelectorAll('#colorForm input[type="checkbox"]').forEach(function(checkbox) {
    checkbox.addEventListener('change', function() {
      // 创建一个空数组来存储选中的颜色
      var selectedColors = [];

      // 遍历所有多选框，检查哪些被选中了
      document.querySelectorAll('#colorForm input[type="checkbox"]:checked').forEach(function(checkedCheckbox) {
        selectedColors.push(checkedCheckbox.value);
      });

      // 显示选中的颜色，如果没有选中任何颜色则显示空字符串
      var hintText = selectedColors.length ? "你选择了 " + selectedColors.join(', ') + "！" : "";
      document.getElementById("hint").textContent = hintText;
    });
  });
</script>

</body>
</html>
```

### Slider
::: code-group
```html [main.html]
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Slider 示例</title>
</head>
<body>

<h2>音量控制</h2>
<input type="range" id="volumeSlider" min="0" max="100" value="50" step="1">
<p>音量: <span id="volumeDisplay">50</span>%</p>

<script src="script.js"></script>
</body>
</html>
```

```js [script.js]
document.getElementById('volumeSlider').addEventListener('input', function() {
  // 获取滑动条的值
  var volume = this.value;

  // 更新页面上显示的值
  document.getElementById('volumeDisplay').textContent = volume;
});
```
:::

## 在markdown中插入html
在Markdown中插入HTML代码是相对直接的。Markdown的设计初衷是易于阅读和编写的纯文本格式，但它支持内嵌HTML代码，这为Markdown文档提供了额外的灵活性和功能性。

要在Markdown中插入HTML，你只需直接将HTML代码放入Markdown文档中即可。Markdown解析器在渲染Markdown文本时，会识别并保留HTML代码，最终这些HTML代码会以HTML的形式展示在渲染后的文档中。

以下是一个简单的示例，展示了如何在Markdown中插入HTML代码：
```html
这是一个Markdown段落。

<p style="color:blue;">这是一个HTML段落，它的文字颜色是蓝色的。</p>

- Markdown列表项
- 另一个Markdown列表项

<ul style="list-style-type:square;">
  <li>HTML列表项，使用方形标记</li>
  <li>另一个HTML列表项</li>
</ul>

Markdown文本继续...
```

::: details 部分渲染效果

这是一个Markdown段落。

<p style="color:blue;">这是一个HTML段落，它的文字颜色是蓝色的。</p>

- Markdown列表项
- 另一个Markdown列表项

<ul style="list-style-type:square;">
  <li>HTML列表项，使用方形标记</li>
  <li>另一个HTML列表项</li>
</ul>
:::

在上述示例中，Markdown解析器会保留`<p>`标签和`<ul>`标签及其内容，并按照HTML的规则来渲染它们。因此，在渲染后的文档中，你会看到一个蓝色的HTML段落和一个使用方形标记的HTML列表，而Markdown部分则会按照Markdown的规则来渲染。

需要注意的是，虽然Markdown支持HTML，但并不是所有的HTML标签都会被Markdown解析器识别或保留。一些HTML标签可能会被Markdown解析器解释为Markdown语法的一部分，从而导致意外的渲染效果。因此，在插入HTML代码时，最好先了解你的Markdown解析器对HTML的支持情况。

## 应用
### 分块视频制作

::: code-group
```html [main.html]
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Video Comparison Slider</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="comparison-container">
    <video id="video1" autoplay loop muted>
      <source src="video1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

    <video id="video2" autoplay loop muted>
      <source src="video2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

    <div class="slider" id="slider"></div>
  </div>

  <script src="script.js"></script>
</body>
</html>
```

```css [style.css]
/* style.css */
.comparison-container {
  position: relative;
  width: 100%;
  max-width: 800px;
  height: 400px;
  margin: auto;
  overflow: hidden;
}

#video1, #video2 {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

#video2 {
  clip-path: inset(0 50% 0 0); /* Initial half of the video */
}

.slider {
  position: absolute;
  top: 0;
  left: 50%;
  width: 5px;
  height: 100%;
  background-color: white;
  cursor: ew-resize;
}
```

```js [script.js]
// script.js
const slider = document.getElementById('slider');
const video2 = document.getElementById('video2');
const container = document.querySelector('.comparison-container');

let isSliding = false;

slider.addEventListener('mousedown', () => {
  isSliding = true;
});

window.addEventListener('mouseup', () => {
  isSliding = false;
});

window.addEventListener('mousemove', (event) => {
  if (isSliding) {
    // Calculate the position of the slider relative to the container
    let rect = container.getBoundingClientRect();
    let position = event.clientX - rect.left;

    // Clamp the position between the left and right edges of the container
    position = Math.max(0, Math.min(position, rect.width));

    // Update slider position and video clipping
    slider.style.left = position + 'px';
    video2.style.clipPath = `inset(0 ${rect.width - position}px 0 0)`;
  }
});
```
:::
