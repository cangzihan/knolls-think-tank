
# C语言

非新手教程

## 数组
### 一维数组
数组定义语句
```c
int a[10];
float b[10], c[20];
char ch[20];

int a[9]={0,1,2,3,4,5,6,7,8};
int a[]={0,1,2,6,7,8};
```

### 二维数组
数组定义语句
```c
int a[2][3]={{1,2,3}, {3,2,1}};
```

### 字符数组
在C语言中没有专门的字符串变量，通常用一个字符数组来存放一个字符串。字符串总是以`'\0'`作为串的结束符。

数组定义语句
```c
char c[]={'C', ' ', 'A', 'P', ‘P’};
char c[]={"C APP"};
char c[]="C APP";
```

字符数组的输入输出

```c
main()
{
  char c[]="BASIC\nBASE";
  printf("%s\n", c)
}
```

```c
main()
{
  char st[15];
  printf("input string:\n");
  scanf("%s", st);
  printf("%s\n", st);
}
```

#### 字符串操作
输出函数`puts`，输入函数`gets`

字符串连接函数`strcat`
```c
#include <stdio.h>
#include"string.h"

main()
{
  static char st1[]="ZS is ";
  int st2[10];
  printf("What is ZS?");
  gets(st2);
  strcat(st1,st2);
  puts(st1);
}

```

字符串复制函数`strcpy`
```c
char st1[15], st2[]="ZSSB";
strcpy(st1, st2);
```

测字符串长度函数`strlen`
```c
int k;
static char st[] = "ZSSB";
k = strlen(st);
```

字符串分割

`strtok`函数会遍历字符串`str`，并在遇到分隔符时将其替换为空字符`\0`，从而生成子字符串。每次调用`strtok`时，它会从上次停止的地方继续分割字符串。

```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello, World! This is a test.";
    const char delim[] = ", !.";

    // 第一次调用 strtok，传入要分割的字符串
    char *token = strtok(str, delim);

    // 循环调用 strtok，传入 NULL 表示继续分割
    while (token != NULL) {
        printf("Token: %s\n", token);
        token = strtok(NULL, delim);
    }

    return 0;
}
```

实现类似于Python的`split`函数
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 100

// 自定义split函数
char** split(const char* str, char delimiter, int* num_tokens) {
    // 分配内存给字符串数组
    char** tokens = (char**)malloc(MAX_TOKENS * sizeof(char*));
    if (tokens == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // 分配内存给临时缓冲区
    char buffer[strlen(str) + 1];
    strcpy(buffer, str);

    // 分割字符串
    char* token = strtok(buffer, &delimiter);
    int index = 0;
    while (token != NULL && index < MAX_TOKENS) {
        tokens[index++] = strdup(token); // 使用strdup复制字符串
        if (tokens[index - 1] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(1);
        }
        token = strtok(NULL, &delimiter);
    }

    *num_tokens = index;
    return tokens;
}

// 主函数测试split函数
int main() {
    const char* input = "gpio,P55,1";
    char delimiter = ',';
    int num_tokens;

    char** tokens = split(input, delimiter, &num_tokens);

    for (int i = 0; i < num_tokens; i++) {
        printf("Token %d: %s\n", i, tokens[i]);
        free(tokens[i]); // 释放每个token的内存
    }

    free(tokens); // 释放tokens数组的内存

    return 0;
}
```

字符串`==`

`strcmp`函数逐个字符地比较两个字符串，直到遇到不同的字符或到达字符串的末尾（即遇到空字符`\0`）。比较规则如下：
1. 逐字符比较：从左到右依次比较两个字符串的对应字符。
2. ASCII 值比较：对于每个字符，`strcmp`比较它们的 ASCII 值。如果`str1`中的字符的 ASCII 值小于`str2`中的字符的 ASCII 值，则返回负数；如果大于，则返回正数；如果相等，则继续比较下一个字符。
3. 结束标志：当遇到空字符`\0`时，比较结束。如果两个字符串都到达末尾，则认为它们相等，返回 0。
```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "gpio";

    // 使用 strcmp 比较字符串
    if (strcmp(str, "gpio") == 0) {
        printf("The string is 'gpio'.\n");
    } else {
        printf("The string is not 'gpio'.\n");
    }

    return 0;
}

```

字符串`in`

`strchr(str, ch)`会在字符串`str`中查找字符`'o'`。如果找到了该字符，`result`将指向该字符在字符串中的位置；如果没有找到，则`result`将为`NULL`。
```c
#include <stdio.h>
#include <string.h>

int main() {
    char str[] = "Hello, World!";
    char ch = 'o';

    // 使用 strchr 查找字符
    char *result = strchr(str, ch);

    if (result != NULL) {
        printf("Character '%c' found at position %ld in the string.\n", ch, result - str);
    } else {
        printf("Character '%c' not found in the string.\n", ch);
    }

    return 0;
}
```


## 条件编译
### 形式1
```c
#ifdef 标识符
  程序段1
#else
  程序段2
#endif
```

```c
#ifndef 标识符
  程序段1
#else
  程序段2
#endif
```

```c
#if 常量表达式
  程序段1
#else
  程序段2
#endif
```

## 指针
### 指针变量
```c
int *p2;
float *p3;
char *p4;
```

```c
main()
{
  int a,b;
  int *p1, *p2;
  a=1, b=98;
  p1 = &a;
  p2 = &b;
  printf("%d,%d\n", a, b);
  printf("%d,%d\n", p1, p2);
}
```

#### 指针变量作为函数参数
```c
swap(int *p1, int *p2)
{
  int temp;
  temp = *p1;
  *p1 = *p2;
  *p2 = temp;
}

main()
{
  int a,b;
  int *pointer_1,*pointer_2;
  scanf("%d,%d", &a,&b);
  pointer_1=&a;
  pointer_2=&b;
  if(a<b)
    swap(pointer_1,pointer_2);
   printf("\n%d,%d\n", a, b)
}

```

### 指向数组的指针
```c
p = &a[0];
p = a;
int *p = &a[0];
```

#### 指向二维数组的指针
```c
main()
{
  int a[3][4]={0,1,2,3,4,5,6,7,8,9,10,11};
  int (*p)[4];
  int i,j;
  p=a;
  for(i=0;i<3;i++)
  {
    for(j=0;j<4;j++)
      printf("%2d ", *(*(p+i)+j));
    printf("\n");
  }
}
```

### 字符串指针
用字符串指针指向一个字符串
```c
char *str="ZSSB";
printf("%s\n", str);
```

### 函数指针
```c
void (*pf)();
```

### 双重指针
```c
main()
{
  char *name[]={"STC89C52RC", "STM32F103C8T6", "RTX3060", "RK3588", "STC8A8KS64D4"};
  char **p;
  int i;
  for(i=0;i<5;i++)
  {
    p = name + i;
    printf("%s\n", *p);
  }
}

```

## 动态存储分配
### 分配内存空间
```c
pc = (char *)malloc(100);
```
表示分配100个字节的内存空间，并强制转换为字符数组类型，函数的返回值为指向该字符数组的指针。

### 释放内存空间
```c
free(*ptr);
```
释放`ptr`所指向的一块内存空间，`ptr`是一个任意类型的指针变量，它指向被释放区域的首地址。
被释放区应是由`malloc`或`calloc`函数所分配的区域。

