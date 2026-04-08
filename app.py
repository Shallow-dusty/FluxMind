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
    st.caption("Sliding Mode Control & Flux Estimation Copilot")
    st.divider()

    st.subheader("📚 Knowledge Base")

    # Show indexed papers
    papers = sorted(PAPERS_DIR.glob("*.pdf")) if PAPERS_DIR.exists() else []
    if papers:
        st.success(f"{len(papers)} papers indexed")
        with st.expander("View papers"):
            for p in papers:
                st.text(f"• {p.name}")
    else:
        st.warning("No papers in knowledge base yet")

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            existing = PAPERS_DIR / uf.name if PAPERS_DIR.exists() else None
            if existing and existing.exists():
                st.info(f"⏭️ {uf.name} already exists, skipping")
                continue
            with st.spinner(f"Indexing {uf.name}..."):
                n_chunks = ingest_uploaded_pdf(uf.read(), uf.name)
                st.success(f"✅ {uf.name} → {n_chunks} chunks")

    # Rebuild index button
    if papers and st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding vector store..."):
            # Delete existing index
            import shutil
            if FAISS_INDEX_DIR.exists():
                shutil.rmtree(FAISS_INDEX_DIR)
            docs = load_all_pdfs()
            build_vector_store(docs)
            st.success("Index rebuilt!")

    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    **FluxMind** is a RAG-based research copilot for control engineering.

    **Capabilities:**
    - 🎯 SMC theory Q&A with citations
    - 🧲 Flux estimation guidance
    - 💻 MATLAB/Simulink code generation
    - 📊 Mathematical derivation support

    **Architecture:**
    `Query → Embedding → FAISS Retrieval → LLM Generation`
    """)

# ── Init vector store on first run ──
if "store_initialized" not in st.session_state:
    with st.spinner("Initializing knowledge base..."):
        build_vector_store()
    st.session_state.store_initialized = True

# ── Chat Interface ──
st.title("⚡ FluxMind")
st.markdown("*Your AI research copilot for Sliding Mode Control & Flux Linkage Estimation*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example questions
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    cols = st.columns(2)
    examples = [
        "Explain the reaching law design in sliding mode control",
        "How to reduce chattering in SMC for motor drives?",
        "Generate MATLAB code for a flux linkage observer using MRAS",
        "Compare EKF and Luenberger observer for PMSM flux estimation",
    ]
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()

# Chat input
if prompt := st.chat_input("Ask about sliding mode control, flux estimation, or MATLAB modeling..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(query_stream(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
