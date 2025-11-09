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

## CSS

### 选择器
它告诉浏览器你想要将样式应用到哪个（些） HTML 元素上。选择器“选择”了你想要应用样式的元素。

#### 基础选择器
分为元素选择器、类选择器、ID选择器和通用选择器

```html
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      h4 {
        color: red;
      }

      .classSelector {
        color: green;
      }

      .idSelector {
        color: blue;
      }
    </style>  
  </head>
  <body>
    <h4>元素选择器</h4>
    <p class="classSelector">类选择器</p>
    <p id="idSelector">ID选择器</p>
  </body>
```

#### 组合器选择器 (Combinator Selectors)
组合器允许你基于元素之间的关系来选择元素。

```css
/* 后代选择器 (Descendant Combinator) */
div p {
  color: green;
}
/* 将所有位于 <div> 元素 内部 的 <p> 元素的文本颜色设置为绿色 */

/* 子选择器 (Child Combinator) */
ul > li {
  list-style-type: square;
}
/* 这会将所有直接位于 <ul> (无序列表) 下的 <li> (列表项) 的符号设置为方块，但不会影响嵌套在 <li> 内部的 <ul> 中的 <li>。 */

/* 相邻兄弟选择器 (Adjacent Sibling Combinator) */
h1 + p {
  font-style: italic;
}
/* 将紧跟在 <h1> 元素 后面 的第一个 <p> 元素的字体设置为斜体。 */

/* 通用兄弟选择器 (General Sibling Combinator) */
h1 ~ p {
  font-weight: bold;
}
/* 将所有位于 <h1> 元素 后面 的 <p> 兄弟元素的字体设置为粗体。 */
```

#### 伪类选择器 (Pseudo-classes)

伪类用于选择元素的特定状态。

*   **`:hover`**
    *   当用户将鼠标悬停在元素上时。
    *   例子：
        ```css
        button:hover {
          background-color: lightgray;
        }
        ```
        当鼠标悬停在按钮上时，背景色变为浅灰色。

*   **`:active`**
    *   当元素被激活（如鼠标点击时）时。

*   **`:focus`**
    *   当元素获得焦点时（如通过 Tab 键或点击输入框）。

*   **`:visited`, `:link`**
    *   分别用于已访问和未访问的链接。

*   **`:first-child`, `:last-child`, `:nth-child(n)`**
    *   选择父元素的第一个、最后一个或第 n 个子元素。
    *   例子：
        ```css
        li:first-child {
          font-weight: bold;
        }
        ```
        将列表中的第一个 `<li>` 项设置为粗体。

*   **`:not(selector)`**
    *   选择不符合括号内选择器条件的元素。
    *   例子：
        ```css
        p:not(.special) {
          color: red;
        }
        ```
        将所有 *不* 具有 `class="special"` 的 `<p>` 元素的文本颜色设置为红色。

#### 伪元素选择器 (Pseudo-elements)

伪元素用于选择元素的特定部分（如首字母、之前或之后的内容）。

*   **`::before`**
    *   在元素内容 *之前* 插入内容。

*   **`::after`**
    *   在元素内容 *之后* 插入内容。

*   **`::first-letter`**
    *   选择元素内容的第一个字母。

*   **`::first-line`**
    *   选择元素内容的第一行。

*   **`::selection`**
    *   选择用户选中的部分。

#### 属性选择器 (Attribute Selectors)

根据元素的属性及其值来选择元素。

*   **`[attribute]`**
    *   选择具有指定属性的元素。
    *   例子：`[title]` 选择所有有 `title` 属性的元素。

*   **`[attribute="value"]`**
    *   选择具有指定属性和值的元素。
    *   例子：`[type="text"]` 选择所有 `type` 属性值为 "text" 的元素。

*   **`[attribute~="value"]`**
    *   选择属性值包含指定单词（以空格分隔）的元素。
    *   例子：`[class~="highlight"]` 会选择 `<div class="main highlight">`，但不会选择 `<div class="highlighted">`。

*   **`[attribute|="value"]`**
    *   选择属性值以指定值开头，且后面紧跟连字符 `-` 的元素（常用于语言代码）。

*   **`[attribute^="value"]`**
    *   选择属性值以指定值 *开头* 的元素。
    *   例子：`[href^="https://"]` 选择所有 `href` 以 "https://" 开头的链接。

*   **`[attribute$="value"]`**
    *   选择属性值以指定值 *结尾* 的元素。
    *   例子：`[href$=".pdf"]` 选择所有 `href` 以 ".pdf" 结尾的链接。

*   **`[attribute*="value"]`**
    *   选择属性值 *包含* 指定值的元素。
    *   例子：`[title*="warning"]` 选择所有 `title` 属性中包含 "warning" 字符串的元素。

#### 选择器优先级 (Specificity)

当多个 CSS 规则应用于同一个元素时，浏览器需要决定应用哪个规则。这由选择器的优先级决定。优先级从高到低通常为：

1.  **内联样式** (直接写在 HTML 元素的 `style` 属性上) - 最高优先级
2.  **ID 选择器** (`#id`)
3.  **类选择器、属性选择器、伪类选择器** (`.class`, `[type="text"]`, `:hover`)
4.  **元素选择器、伪元素选择器** (`div`, `::before`)
5.  **通用选择器** (`*`) - 最低优先级

如果两个规则具有相同的选择器优先级，则后定义的规则会覆盖前面的规则。`!important` 可以覆盖所有优先级规则，但应谨慎使用。

### 文本
#### 字体
```css
p {
  font-family: Arial, sans-serif; /* 优先使用 Arial，如果不可用则使用系统默认的无衬线字体 */

  /* 设置字体的大小*/
  font-size: 2em; /* 相对于父元素字体大小的2倍 */
  /* font-size: 16px; /* 绝对像素值 */
  /* font-size: 1.2rem; /* 相对于根元素 (html) 字体大小 */

  /* 设置字体的粗细程度*/
  font-weight: bold; /* 或者 font-weight: 700; */

  font-style: italic; /* normal, italic, oblique */
}
```

简写属性
```css
p {
  font: italic bold 16px/1.4 Arial, sans-serif; /* 样式 粗细 大小/行高 字体族 */
}
```

#### 文本外观
```css
a:link {
  color: blue;
}

a:visited {
  text-decoration: underline line-through; /* 同时有下划线和删除线 */
}

h1 {
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* 水平偏移2px，垂直偏移2px，模糊半径4px，半透明黑色阴影 */
}
```

#### 文本布局

这些属性控制文本在行和块中的排列方式。

*   **`text-align`**
    *   **作用**: 设置行内内容（如文本）在其包含块中的水平对齐方式。
    *   **值**: `left`, `right`, `center`, `justify` (两端对齐)。
    *   **示例**:
        ```css
        p {
          text-align: justify; /* 使文本左右两端都对齐 */
        }
        ```

*   **`text-indent`**
    *   **作用**: 设置块级元素（如段落）第一行的缩进。
    *   **示例**:
        ```css
        p {
          text-indent: 2em; /* 第一行缩进两个字符的宽度 */
        }
        ```

*   **`line-height`**
    *   **作用**: 设置行与行之间的距离（行高）。
    *   **示例**:
        ```css
        p {
          line-height: 1.6; /* 设置为字体大小的1.6倍 */
        }
        ```

*   **`letter-spacing`**
    *   **作用**: 设置字符（字母、数字、符号等）之间的间距。
    *   **示例**:
        ```css
        .spaced-out {
          letter-spacing: 2px; /* 每个字符之间增加2px间距 */
        }
        ```

*   **`word-spacing`**
    *   **作用**: 设置单词之间的间距。
    *   **示例**:
        ```css
        .wide-words {
          word-spacing: 10px; /* 单词之间增加10px间距 */
        }
        ```

*   **`vertical-align`**
    *   **作用**: 控制行内元素（如图片、`<span>`）或表格单元格内容的垂直对齐方式。*（注意：它对块级元素的对齐无效）*
    *   **值**: `baseline`, `top`, `middle`, `bottom`, `text-top`, `text-bottom`, `<length>`, `<percentage>`。
    *   **示例**:
        ```css
        img {
          vertical-align: middle; /* 使图片与文本基线对齐 */
        }
        ```

#### 文本溢出 (Text Overflow) 属性

这些属性控制当文本内容超出容器边界时的处理方式。

*   **`text-overflow`**
    *   **作用**: 指定当文本溢出其包含块时如何显示（例如，显示省略号 `...`）。
    *   **值**: `clip` (裁剪), `ellipsis` (显示省略号)。
    *   **注意**: 通常需要配合 `white-space: nowrap` 和 `overflow: hidden` 使用。
    *   **示例**:
        ```css
        .truncate {
          width: 200px; /* 限制容器宽度 */
          overflow: hidden; /* 隐藏溢出内容 */
          white-space: nowrap; /* 防止文本换行 */
          text-overflow: ellipsis; /* 溢出时显示省略号 */
        }
        ```

#### 白色空间 (Whitespace) 属性

*   **`white-space`**
    *   **作用**: 指定如何处理元素内的空白字符（空格、换行符、制表符等）。
    *   **值**:
        *   `normal`: 合并空白字符和换行符，必要时换行。
        *   `nowrap`: 合并空白字符，但不换行。
        *   `pre`: 保留空白字符和换行符，不自动换行（类似 `<pre>` 标签）。
        *   `pre-wrap`: 保留空白字符和换行符，但必要时换行。
        *   `pre-line`: 合并空白字符，但保留换行符，必要时换行。
    *   **示例**:
        ```css
        .preformatted {
          white-space: pre-wrap; /* 保留代码或文本的格式 */
        }
        ```

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

### 分块图片制作

::: code-group
```html [main.html]
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Comparison Slider</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="comparison-container">
    <img src="image1.jpg" id="image1" alt="Before">
    <img src="image2.jpg" id="image2" alt="After">
    <div class="slider" id="slider"></div>
  </div>

  <script src="script.js"></script>
</body>
</html>

```

```css [style.css]
.comparison-container {
  position: relative;
  width: 100%;
  max-width: 2000px;
  height: 3000px; /* 确保容器有固定高度 */
  margin: auto;
  overflow: hidden;
}

#image1, #image2 {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover; /* 让图片填满容器，不变形 */
  user-select: none; /* 禁止用户选中 */
  pointer-events: none; /* 禁止鼠标事件，避免拖动时误选 */
}


#image2 {
  clip-path: inset(0 50% 0 0); /* 初始显示左半部分 */
}

.slider {
  position: absolute;
  top: 0;
  left: 50%;
  width: 5px;
  height: 100%;
  background-color: white;
  cursor: ew-resize;
  transform: translateX(-50%);
}

```

```js [script.js]
const slider = document.getElementById('slider');
const image2 = document.getElementById('image2');
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
    let rect = container.getBoundingClientRect();
    let position = event.clientX - rect.left;

    position = Math.max(0, Math.min(position, rect.width));

    // 更新滑块位置
    slider.style.left = position + 'px';

    // 使用 clip-path 控制 image2 的可见区域
    image2.style.clipPath = `inset(0 ${rect.width - position}px 0 0)`;
  }
});
```
:::
