# ⚡ FluxMind

**A RAG-based Research Copilot for Sliding Mode Control & Flux Linkage Estimation**

FluxMind is an intelligent research assistant built on Retrieval-Augmented Generation (RAG) architecture, designed to help control engineering researchers and students with theoretical analysis, MATLAB code generation, and literature navigation in the domains of Sliding Mode Control (SMC) and Flux Linkage Estimation.

## ✨ Features

- **📖 Literature-Grounded Q&A** — Ask theoretical questions and get answers with citations from your uploaded papers
- **💻 MATLAB Code Generation** — Generate control system code (observers, controllers, Simulink blocks) on demand
- **🧲 Domain Expertise** — Specialized in SMC reaching law design, chattering reduction, MRAS observers, EKF-based estimation
- **📊 Mathematical Support** — LaTeX-formatted equations and derivations
- **🌐 Bilingual** — Supports both English and Chinese queries

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│              (Chat UI / PDF Upload / History)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Intent & Query Layer                     │
│                                                           │
│  User Query ──► Embedding ──► Vector Similarity Search    │
│                  (local)        (FAISS, top-k=5)          │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────────────────┐
│  Vector Database  │  │        LLM Generation Layer       │
│  (FAISS Index)    │  │                                    │
│                   │  │  Retrieved Context + System Prompt │
│  PDF ► Chunks ►   │  │         ▼                          │
│  Embeddings ►     │──│  DeepSeek-V3.2 (via OpenAI API)   │
│  Index            │  │         ▼                          │
│                   │  │  Cited Answer / MATLAB Code        │
└──────────────────┘  └──────────────────────────────────┘
```

### Data Pipeline

```
PDF Papers ──► PyMuPDF (text extraction)
           ──► RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
           ──► SentenceTransformer (all-MiniLM-L6-v2, local embedding)
           ──► FAISS Index (L2 similarity, persistent storage)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or venv

### Installation

```bash
# Clone the repository
git clone https://github.com/Shallow-dusty/FluxMind.git
cd FluxMind

# Create environment
conda create -n fluxmind python=3.11 -y
conda activate fluxmind

# Install dependencies
pip install -r requirements.txt

# Configure API
cp .env.example .env
# Edit .env with your LLM API credentials
```

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Docker

```bash
docker build -t fluxmind .
docker run -p 8501:8501 --env-file .env fluxmind
```

## 📚 Building the Knowledge Base

1. Place PDF research papers in the `papers/` directory, or upload them through the web interface
2. The system automatically processes PDFs: extraction → chunking → embedding → indexing
3. Use the "Rebuild Index" button in the sidebar to re-index all papers

### Recommended Paper Topics

- Sliding Mode Control (SMC) fundamentals and reaching law design
- Higher-order SMC and super-twisting algorithms
- Chattering reduction techniques
- Flux linkage observers (MRAS, Luenberger, EKF)
- PMSM/IM sensorless control
- Adaptive and robust observer design

## 🔧 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| RAG Framework | LangChain | Orchestration of retrieval and generation pipeline |
| Vector Store | FAISS | Local vector similarity search (no external DB needed) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local text embedding (384-dim) |
| LLM | DeepSeek-V3.2 | Response generation via OpenAI-compatible API |
| PDF Parser | PyMuPDF (fitz) | Fast and accurate PDF text extraction |
| Frontend | Streamlit | Interactive chat interface with file upload |

## 📁 Project Structure

```
FluxMind/
├── app.py                 # Streamlit application entry point
├── src/
│   ├── config.py          # Configuration and environment variables
│   ├── embeddings.py      # Local embedding model setup
│   ├── ingestion.py       # PDF loading, chunking, and indexing
│   └── chain.py           # RAG chain (retrieval + LLM generation)
├── papers/                # Research paper PDFs (git-ignored)
├── faiss_index/           # Persistent FAISS index (git-ignored)
├── assets/                # Architecture diagrams and images
├── docs/                  # Additional documentation
├── Dockerfile             # Container deployment
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variable template
```

## 📄 License

MIT
