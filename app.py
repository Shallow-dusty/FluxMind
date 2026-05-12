"""FluxMind — RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation."""

import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="FluxMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.chain import query_stream
from src.ingestion import ingest_uploaded_pdf, build_vector_store, load_all_pdfs, PAPERS_DIR, FAISS_INDEX_DIR

I18N = {
    "zh": {
        "language": "语言",
        "caption": "滑模控制与磁链估计研究助手",
        "knowledge_base": "知识库",
        "papers_indexed": "{count} 篇论文已索引",
        "view_papers": "查看论文",
        "no_papers": "知识库暂无论文",
        "upload_papers": "上传研究论文（PDF）",
        "already_exists": "{filename} 已存在，已跳过",
        "indexing": "正在索引 {filename}...",
        "indexed_chunks": "{filename} -> {chunks} 个文本块",
        "rebuild_index": "重建索引",
        "rebuilding": "正在重建向量库...",
        "rebuilt": "索引已重建",
        "about": "关于",
        "about_text": """
        **FluxMind** 是面向控制工程的 RAG 研究助手。

        **能力：**
        - 滑模控制理论问答与引用
        - 磁链估计方案梳理
        - MATLAB/Simulink 代码生成
        - 数学推导与公式说明

        **架构：**
        `问题 -> Embedding -> FAISS 检索 -> LLM 生成`
        """,
        "initializing": "正在初始化知识库...",
        "hero_subtitle": "*你的滑模控制与磁链估计 AI 研究助手*",
        "try_asking": "### 试试这些问题：",
        "examples": [
            "解释滑模控制中的趋近律设计",
            "电机驱动中的 SMC 如何削弱抖振？",
            "生成一个基于 MRAS 的磁链观测器 MATLAB 示例",
            "对比 PMSM 磁链估计中的 EKF 与 Luenberger 观测器",
        ],
        "chat_placeholder": "询问滑模控制、磁链估计或 MATLAB 建模问题...",
    },
    "en": {
        "language": "Language",
        "caption": "Sliding Mode Control & Flux Estimation Copilot",
        "knowledge_base": "Knowledge Base",
        "papers_indexed": "{count} papers indexed",
        "view_papers": "View papers",
        "no_papers": "No papers in knowledge base yet",
        "upload_papers": "Upload research papers (PDF)",
        "already_exists": "{filename} already exists, skipping",
        "indexing": "Indexing {filename}...",
        "indexed_chunks": "{filename} -> {chunks} chunks",
        "rebuild_index": "Rebuild Index",
        "rebuilding": "Rebuilding vector store...",
        "rebuilt": "Index rebuilt!",
        "about": "About",
        "about_text": """
        **FluxMind** is a RAG-based research copilot for control engineering.

        **Capabilities:**
        - SMC theory Q&A with citations
        - Flux estimation guidance
        - MATLAB/Simulink code generation
        - Mathematical derivation support

        **Architecture:**
        `Query -> Embedding -> FAISS Retrieval -> LLM Generation`
        """,
        "initializing": "Initializing knowledge base...",
        "hero_subtitle": "*Your AI research copilot for Sliding Mode Control & Flux Linkage Estimation*",
        "try_asking": "### Try asking:",
        "examples": [
            "Explain the reaching law design in sliding mode control",
            "How to reduce chattering in SMC for motor drives?",
            "Generate MATLAB code for a flux linkage observer using MRAS",
            "Compare EKF and Luenberger observer for PMSM flux estimation",
        ],
        "chat_placeholder": "Ask about sliding mode control, flux estimation, or MATLAB modeling...",
    },
}

# ── Custom CSS ──
st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .source-tag {
        display: inline-block;
        background: #e8f4f8;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.8em;
        color: #1a5276;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: Knowledge Base Management ──
with st.sidebar:
    st.title("⚡ FluxMind")
    language = st.selectbox(
        "语言 / Language",
        options=["zh", "en"],
        format_func=lambda value: "中文" if value == "zh" else "English",
        index=0,
        key="language",
    )
    text = I18N[language]
    st.caption(text["caption"])
    st.divider()

    st.subheader(f"📚 {text['knowledge_base']}")

    # Show indexed papers
    papers = sorted(PAPERS_DIR.glob("*.pdf")) if PAPERS_DIR.exists() else []
    if papers:
        st.success(text["papers_indexed"].format(count=len(papers)))
        with st.expander(text["view_papers"]):
            for p in papers:
                st.text(f"• {p.name}")
    else:
        st.warning(text["no_papers"])

    # Upload PDFs
    uploaded_files = st.file_uploader(
        text["upload_papers"],
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            existing = PAPERS_DIR / uf.name if PAPERS_DIR.exists() else None
            if existing and existing.exists():
                st.info(text["already_exists"].format(filename=uf.name))
                continue
            with st.spinner(text["indexing"].format(filename=uf.name)):
                n_chunks = ingest_uploaded_pdf(uf.read(), uf.name)
                st.success(text["indexed_chunks"].format(filename=uf.name, chunks=n_chunks))

    # Rebuild index button
    if papers and st.button(f"🔄 {text['rebuild_index']}", use_container_width=True):
        with st.spinner(text["rebuilding"]):
            # Delete existing index
            import shutil
            if FAISS_INDEX_DIR.exists():
                shutil.rmtree(FAISS_INDEX_DIR)
            docs = load_all_pdfs()
            build_vector_store(docs)
            st.success(text["rebuilt"])

    st.divider()
    st.subheader(f"ℹ️ {text['about']}")
    st.markdown(text["about_text"])

# ── Init vector store on first run ──
if "store_initialized" not in st.session_state:
    with st.spinner(text["initializing"]):
        build_vector_store()
    st.session_state.store_initialized = True

# ── Chat Interface ──
st.title("⚡ FluxMind")
st.markdown(text["hero_subtitle"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example questions
if not st.session_state.messages:
    st.markdown(text["try_asking"])
    cols = st.columns(2)
    for i, ex in enumerate(text["examples"]):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

# Chat input
if prompt := st.chat_input(text["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(query_stream(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
