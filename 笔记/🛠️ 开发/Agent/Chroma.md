---
tags:
  - 数据库
---

# Chroma
Chroma 是一个开源的向量数据库，专为 AI 应用程序设计，特别适用于检索增强生成 (RAG)、语义搜索和相似性搜索等场景。

[Doc](https://docs.trychroma.com/docs/overview/getting-started)

核心特性
1. 向量存储
    - 高效存储和检索向量嵌入
    - 支持高维向量数据
    - 优化的索引结构
2. 相似性搜索
    - 快速的最近邻搜索
    - 支持余弦相似度、欧几里得距离等多种距离度量
    - 实时相似性匹配
3. 元数据支持
    - 丰富的元数据存储
    - 支持过滤和分面搜索
    - 结构化数据关联

相关工具：
- https://github.com/msteele3/chromagraphic
- https://github.com/AYOUYI/chromadb_GUI_tool

## Docker部署
使用`chromadb/chroma`镜像
