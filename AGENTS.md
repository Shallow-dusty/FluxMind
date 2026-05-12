# FluxMind

基于 RAG 的控制理论研究 Copilot（滑模控制 + 磁链估计）。

## 技术栈

- **RAG 框架**: LangChain
- **向量存储**: FAISS (本地)
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2, 本地)
- **LLM**: DeepSeek-V3.2 via OpenAI-compatible API
- **前端**: Streamlit
- **PDF 解析**: PyMuPDF

## 目录结构

```
src/           # 核心模块 (config, embeddings, ingestion, chain)
papers/        # PDF 论文存放 (gitignored)
faiss_index/   # FAISS 向量索引 (gitignored)
assets/        # 架构图等静态资源
docs/          # 项目文档
app.py         # Streamlit 入口
```

## 运行

```bash
conda activate fluxmind
streamlit run app.py
```

## Linear Project

无（临时项目，不走 Linear）
