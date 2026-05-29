"""FluxMind — RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation."""

import streamlit as st
import streamlit.components.v1 as components

# Must be first Streamlit call
st.set_page_config(
    page_title="FluxMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.chain import query_stream
from src.config import MAX_UPLOAD_SIZE_MB, PAPERS_LIBRARY_DIR, PROJECT_ROOT
from src.ingestion import (
    build_vector_store,
    discover_pdfs,
    ingest_uploaded_pdf,
    load_active_paper_paths,
    load_library_manifest,
    rebuild_vector_store_from_pdfs,
)

DEMO_SCRIPT_PATH = PROJECT_ROOT / "docs" / "demo-script.md"


@st.dialog("演示导览", width="large")
def show_demo_guide():
    """Presenter-only walkthrough. Sourced from docs/demo-script.md so the
    content can be edited without touching app code."""
    try:
        content = DEMO_SCRIPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        content = "_`docs/demo-script.md` not found._"
    st.markdown(content)

I18N = {
    "zh": {
        "language": "语言",
        "caption": "滑模控制与磁链估计研究助手",
        "knowledge_base": "知识库",
        "papers_indexed": "{count} 篇论文已索引",
        "available_papers": "可选论文库",
        "selected_papers": "当前入库论文",
        "select_papers": "选择要进入 RAG 的论文",
        "apply_selection": "应用选择并重建索引",
        "no_selectable_papers": "还没有可选 PDF",
        "view_papers": "查看论文",
        "no_papers": "知识库暂无论文",
        "upload_papers": "上传研究论文（PDF）",
        "already_exists": "{filename} 已存在，已跳过",
        "indexing": "正在索引 {filename}...",
        "indexed_chunks": "{filename} -> {chunks} 个文本块",
        "rebuild_index": "重建索引",
        "rebuilding": "正在重建向量库...",
        "rebuilt": "索引已重建",
        "select_at_least_one": "请至少选择一篇论文",
        "upload_failed": "上传失败：{error}",
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
        "available_papers": "Selectable Library",
        "selected_papers": "Active papers",
        "select_papers": "Choose papers for RAG",
        "apply_selection": "Apply Selection and Rebuild Index",
        "no_selectable_papers": "No selectable PDFs yet",
        "view_papers": "View papers",
        "no_papers": "No papers in knowledge base yet",
        "upload_papers": "Upload research papers (PDF)",
        "already_exists": "{filename} already exists, skipping",
        "indexing": "Indexing {filename}...",
        "indexed_chunks": "{filename} -> {chunks} chunks",
        "rebuild_index": "Rebuild Index",
        "rebuilding": "Rebuilding vector store...",
        "rebuilt": "Index rebuilt!",
        "select_at_least_one": "Select at least one paper",
        "upload_failed": "Upload failed: {error}",
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
<meta name="google" content="notranslate">
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
    /* Decorative sidebar footer egg — faded by default, lights up on hover */
    .st-key-bg_easter button {
        opacity: 0.22;
        border-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        transition: opacity .35s ease, background .35s ease;
    }
    .st-key-bg_easter button:hover {
        opacity: 1;
        background: rgba(77, 61, 166, 0.08) !important;
    }
    .st-key-bg_easter button p { font-size: 18px; line-height: 1; }
</style>
""", unsafe_allow_html=True)


def install_translation_guard() -> None:
    """Mark the Streamlit document as non-translatable.

    Browser translation extensions can mutate text nodes while Streamlit is
    streaming markdown into the same container. That can leave the frontend's
    virtual DOM out of sync with the real DOM and break streamed rendering.
    """
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const mark = (node) => {
            if (!node || !node.setAttribute) return;
            node.setAttribute("translate", "no");
            node.classList.add("notranslate");
        };
        mark(doc.documentElement);
        mark(doc.body);
        doc
          .querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stChatMessage"]')
          .forEach(mark);
        new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        mark(node);
                        node
                          .querySelectorAll?.('[data-testid="stChatMessage"], .stMarkdown')
                          .forEach(mark);
                    }
                }
            }
        }).observe(doc.body, { childList: true, subtree: true });
        </script>
        """,
        height=0,
        width=0,
    )


install_translation_guard()


def rel_path(path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def paper_label(path, manifest: dict[str, dict]) -> str:
    item = manifest.get(path.name, {})
    title = item.get("title") or path.stem.replace("-", " ")
    topic = item.get("topic")
    source = "Seed" if PAPERS_LIBRARY_DIR in path.parents else "Upload"
    return f"[{source}] {title}" + (f" · {topic}" if topic else "")


def render_streaming_response(prompt: str) -> str:
    """Render a streaming answer through a stable markdown placeholder."""
    chunks: list[str] = []
    placeholder = st.empty()
    for piece in query_stream(prompt):
        chunks.append(piece)
        placeholder.markdown("".join(chunks))
    return "".join(chunks)


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

    manifest = load_library_manifest()
    selectable_papers = discover_pdfs()
    active_papers = load_active_paper_paths()
    selectable_by_rel = {rel_path(path): path for path in selectable_papers}
    active_defaults = [
        rel_path(path) for path in active_papers if rel_path(path) in selectable_by_rel
    ]

    if selectable_papers:
        st.success(text["papers_indexed"].format(count=len(active_defaults)))
        selected = st.multiselect(
            text["select_papers"],
            options=list(selectable_by_rel),
            default=active_defaults,
            format_func=lambda key: paper_label(selectable_by_rel[key], manifest),
            key="paper_selection",
        )
        if st.button(text["apply_selection"], use_container_width=True):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                with st.spinner(text["rebuilding"]):
                    paths = [selectable_by_rel[key] for key in selected]
                    _, chunks = rebuild_vector_store_from_pdfs(paths)
                    st.success(f"{text['rebuilt']} ({chunks} chunks)")
                    st.rerun()
        with st.expander(text["view_papers"]):
            for p in selectable_papers:
                marker = "✓" if rel_path(p) in active_defaults else " "
                st.text(f"{marker} {paper_label(p, manifest)}")
    else:
        st.warning(text["no_selectable_papers"])

    # Upload PDFs
    uploaded_files = st.file_uploader(
        text["upload_papers"],
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(text["indexing"].format(filename=uf.name)):
                try:
                    saved_path, n_chunks = ingest_uploaded_pdf(uf.read(), uf.name)
                    st.success(text["indexed_chunks"].format(filename=saved_path.name, chunks=n_chunks))
                except ValueError as exc:
                    st.error(text["upload_failed"].format(error=exc))

    st.caption(f"Max upload: {MAX_UPLOAD_SIZE_MB} MB")

    st.divider()
    st.subheader(f"ℹ️ {text['about']}")
    st.markdown(text["about_text"])

    # Decorative footer egg — presenter-only demo walkthrough.
    if st.button("🥚", key="bg_easter", help=None):
        show_demo_guide()

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
        response = render_streaming_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
