"""FluxMind — RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation."""

import json
import sqlite3

import streamlit as st

# Must be first Streamlit call
st.set_page_config(
    page_title="FluxMind",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.admin import (
    apply_retention_delete,
    collect_admin_status,
    collect_corpus_profile_status,
    collect_retention_preview,
    format_admin_metrics,
    format_admin_status_report,
    format_corpus_profile_status_report,
)
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.artifacts import (
    LocalArtifactRegistry,
    artifact_to_dict,
    job_artifact_to_dict,
    local_artifact_path,
    safe_artifact_download_filename,
)
from src.chain import query_stream
from src.config import (
    CODE_EXECUTION_BACKEND,
    DOCKER_OCTAVE_EXECUTION_IMAGE,
    DOCKER_PYTHON_EXECUTION_IMAGE,
    IMAGE_PROVIDER_BACKEND,
    MAX_UPLOAD_SIZE_MB,
    OPENAI_IMAGE_API_KEY,
    OPENAI_IMAGE_MODEL,
    PAPERS_LIBRARY_DIR,
    PROJECT_ROOT,
)
from src.execution_templates import OCTAVE_EXECUTION_TEMPLATES, PYTHON_EXECUTION_TEMPLATES
from src.ingestion import (
    build_vector_store,
    discover_pdfs,
    ingest_uploaded_pdf,
    load_active_paper_paths,
    load_library_manifest,
    rebuild_vector_store_from_pdfs,
    set_active_paper_source_paths,
)
from src.jobs import LocalJobRunner, LocalJobStore, get_async_job_manager, job_view
from src.metadata import CorpusProfileStore, safe_corpus_profile_report_filename
from src.runtime import (
    list_runtime_events,
    logger,
    new_request_id,
    normalize_exception,
    runtime_event_to_dict,
)
from src.storage_manifest import (
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
    format_runtime_backup_manifest_markdown,
    format_runtime_restore_check_markdown,
)
from src.users import LocalUserStore, UserAccount
from scripts._cli import format_error

DEMO_SCRIPT_PATH = PROJECT_ROOT / "docs" / "demo-script.md"

RUNTIME_EVENT_KIND_FILTER_OPTIONS = (
    "",
    "provider_failure",
    "provider_quota_guard",
    "query_usage",
    "retrieval_trace",
    "code_execution",
    "admin_check",
    "upload_scan",
    "retention_delete",
)


def streamlit_error_message(exc: BaseException) -> str:
    return format_error(exc)


def streamlit_error_text(template: str, exc: BaseException) -> str:
    return template.format(error=streamlit_error_message(exc))


def streamlit_status_message(message: object, *, fallback: str) -> str:
    return str(message or "").strip() or fallback


def account_ownership(user: UserAccount) -> dict[str, str]:
    """Attach Streamlit jobs to the signed-in local account."""
    return {
        "owner_id": user.user_id,
        "owner_label": user.display_name,
        "ownership_source": "streamlit_user",
    }


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
        "save_selection": "仅保存启用状态",
        "selection_saved": "启用状态已保存；如需立即更新检索范围，请重建索引",
        "corpus_profiles": "语料配置",
        "profile_name": "配置名称",
        "save_profile": "保存当前选择为配置",
        "activate_profile": "启用配置",
        "activate_profile_rebuild": "启用配置并以任务重建索引",
        "download_profile_report": "下载配置状态报告",
        "profile_report_failed": "配置报告生成失败：{error}",
        "profile_saved": "语料配置已保存",
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
        "jobs": "本地任务",
        "latest_jobs": "最近任务",
        "job_search": "任务搜索",
        "job_status_filter": "任务状态",
        "job_kind_filter": "任务类型",
        "cancel_job": "取消任务",
        "retry_job": "重试任务",
        "schedule_retry": "延迟重试",
        "latest_artifacts": "最近产物",
        "artifact_search": "产物搜索",
        "artifact_kind_filter": "产物类型",
        "artifact_job_filter": "任务类型",
        "no_artifacts": "暂无产物",
        "download_artifact": "下载产物",
        "artifact_id": "产物 ID",
        "artifact_metadata": "产物元数据",
        "admin_status": "运行状态",
        "refresh_status": "刷新状态",
        "download_status_report": "下载状态报告",
        "download_metrics": "下载指标文本",
        "download_runtime_manifest": "下载运行时备份清单",
        "download_runtime_restore_check": "下载恢复校验报告",
        "runtime_restore_manifest_upload": "上传运行时备份清单 JSON",
        "runtime_restore_check": "恢复校验",
        "runtime_restore_invalid_manifest": "清单解析失败：{error}",
        "retention_preview": "保留预览",
        "upload_retention_days": "上传保留天数",
        "artifact_retention_days": "产物保留天数",
        "retention_limit": "候选上限",
        "retention_uploads": "上传候选",
        "retention_artifacts": "产物候选",
        "retention_delete": "删除候选",
        "retention_delete_result": "删除结果",
        "runtime_events": "运行事件",
        "event_kind_filter": "事件类型",
        "event_code_filter": "事件代码",
        "event_search": "事件搜索",
        "status_jobs": "任务",
        "status_artifacts": "产物",
        "status_corpus": "语料",
        "status_provider_failures": "Provider 错误",
        "status_query_usage": "查询用量估算",
        "status_retrieval_traces": "检索追踪",
        "status_cost_pricing": "成本估算配置",
        "status_code_execution": "代码执行事件",
        "status_admin_checks": "Admin 检查事件",
        "status_upload_scan": "上传扫描",
        "status_execution_policy": "执行策略",
        "status_storage_inventory": "本地存储盘点",
        "status_runtime_manifest": "运行时备份清单",
        "status_runtime_dirs": "运行目录",
        "no_jobs": "暂无任务",
        "run_index_job": "以任务重建索引",
        "mock_image_job": "生成图像（当前后端）",
        "mock_image_prompt": "图示提示词",
        "mock_image_template": "图示模板",
        "image_backend_caption": "当前图像后端：{backend}",
        "image_model_caption": "OpenAI 图像模型：{model}",
        "image_missing_key": "当前图像后端为 OpenAI，但尚未配置 OPENAI_IMAGE_API_KEY 或 OPENAI_API_KEY。",
        "run_mock_image": "运行图像任务",
        "python_job": "运行 Python（当前后端）",
        "code_execution_backend_caption": "当前代码执行后端：{backend}",
        "code_execution_image_caption": "Docker 镜像：{image}",
        "python_entrypoint": "入口文件",
        "python_template": "Python 模板",
        "python_files": "文件内容",
        "run_python_job": "运行 Python 任务",
        "octave_job": "运行 Octave 兼容脚本（当前后端）",
        "octave_entrypoint": "Octave 入口文件",
        "octave_template": "Octave 模板",
        "octave_files": "Octave 文件内容",
        "run_octave_job": "运行 Octave 任务",
        "execution_templates": {
            "hello": "最小输出",
            "smc_reaching_law": "SMC 趋近律响应",
            "pmsm_current_step": "PMSM q 轴电流阶跃",
            "pmsm_current_decay": "PMSM 电流响应",
            "smc_sign_switching": "SMC 符号切换",
        },
        "answer_mode": "回答模式",
        "signed_in_as": "当前用户：{name}（{role}）",
        "logout": "退出登录",
        "query_history": "查询历史",
        "no_query_history": "暂无查询历史",
        "load_history": "载入",
        "clear_history": "清空我的历史",
        "account_management": "用户管理",
        "create_user": "创建用户",
        "user_id": "用户 ID",
        "display_name": "显示名称",
        "password": "密码",
        "role": "角色",
        "active": "启用",
        "save_user": "保存用户",
        "reset_password": "重置密码",
        "user_created": "用户已创建",
        "user_saved": "用户已更新",
        "password_reset": "密码已更新",
        "answer_modes": {
            "explanation": "解释",
            "derivation": "推导",
            "implementation": "实现",
            "literature_review": "文献综述",
            "code_generation": "代码生成",
        },
        "job_created": "任务状态：{job_id} ({status})",
        "job_failed": "任务失败：{message}",
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
        "initialization_failed": "知识库不可用：{error} 请先由管理员上传并索引 PDF。",
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
        "save_selection": "Save Active State Only",
        "selection_saved": "Active state saved; rebuild the index to update retrieval scope",
        "corpus_profiles": "Corpus profiles",
        "profile_name": "Profile name",
        "save_profile": "Save Current Selection as Profile",
        "activate_profile": "Activate Profile",
        "activate_profile_rebuild": "Activate Profile and Rebuild as Job",
        "download_profile_report": "Download Profile Status Report",
        "profile_report_failed": "Profile report failed: {error}",
        "profile_saved": "Corpus profile saved",
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
        "jobs": "Local Jobs",
        "latest_jobs": "Latest jobs",
        "job_search": "Job search",
        "job_status_filter": "Job status",
        "job_kind_filter": "Job kind",
        "cancel_job": "Cancel job",
        "retry_job": "Retry job",
        "schedule_retry": "Schedule retry",
        "latest_artifacts": "Latest artifacts",
        "artifact_search": "Artifact search",
        "artifact_kind_filter": "Artifact kind",
        "artifact_job_filter": "Job kind",
        "no_artifacts": "No artifacts yet",
        "download_artifact": "Download artifact",
        "artifact_id": "Artifact ID",
        "artifact_metadata": "Artifact metadata",
        "admin_status": "Runtime status",
        "refresh_status": "Refresh status",
        "download_status_report": "Download status report",
        "download_metrics": "Download metrics text",
        "download_runtime_manifest": "Download runtime manifest",
        "download_runtime_restore_check": "Download restore check report",
        "runtime_restore_manifest_upload": "Upload runtime manifest JSON",
        "runtime_restore_check": "Restore check",
        "runtime_restore_invalid_manifest": "Manifest parse failed: {error}",
        "retention_preview": "Retention preview",
        "upload_retention_days": "Upload retention days",
        "artifact_retention_days": "Artifact retention days",
        "retention_limit": "Candidate limit",
        "retention_uploads": "Upload candidates",
        "retention_artifacts": "Artifact candidates",
        "retention_delete": "Delete candidates",
        "retention_delete_result": "Delete result",
        "runtime_events": "Runtime events",
        "event_kind_filter": "Event kind",
        "event_code_filter": "Event code",
        "event_search": "Event search",
        "status_jobs": "Jobs",
        "status_artifacts": "Artifacts",
        "status_corpus": "Corpus",
        "status_provider_failures": "Provider failures",
        "status_query_usage": "Query usage estimates",
        "status_retrieval_traces": "Retrieval traces",
        "status_cost_pricing": "Cost estimate pricing",
        "status_code_execution": "Code execution events",
        "status_admin_checks": "Admin check events",
        "status_upload_scan": "Upload scan",
        "status_execution_policy": "Execution policy",
        "status_storage_inventory": "Local storage inventory",
        "status_runtime_manifest": "Runtime backup manifest",
        "status_runtime_dirs": "Runtime directories",
        "no_jobs": "No jobs yet",
        "run_index_job": "Rebuild Index as Job",
        "mock_image_job": "Generate Image (configured backend)",
        "mock_image_prompt": "Diagram prompt",
        "mock_image_template": "Diagram template",
        "image_backend_caption": "Current image backend: {backend}",
        "image_model_caption": "OpenAI image model: {model}",
        "image_missing_key": "Current image backend is OpenAI, but OPENAI_IMAGE_API_KEY or OPENAI_API_KEY is not configured.",
        "run_mock_image": "Run Image Job",
        "python_job": "Run Python (configured backend)",
        "code_execution_backend_caption": "Current code execution backend: {backend}",
        "code_execution_image_caption": "Docker image: {image}",
        "python_entrypoint": "Entrypoint",
        "python_template": "Python template",
        "python_files": "File contents",
        "run_python_job": "Run Python Job",
        "octave_job": "Run Octave-Compatible Script (configured backend)",
        "octave_entrypoint": "Octave entrypoint",
        "octave_template": "Octave template",
        "octave_files": "Octave file contents",
        "run_octave_job": "Run Octave Job",
        "execution_templates": {
            "hello": "Minimal output",
            "smc_reaching_law": "SMC reaching-law response",
            "pmsm_current_step": "PMSM q-axis current step",
            "pmsm_current_decay": "PMSM current response",
            "smc_sign_switching": "SMC sign switching",
        },
        "answer_mode": "Answer Mode",
        "signed_in_as": "Signed in as {name} ({role})",
        "logout": "Sign out",
        "query_history": "Query history",
        "no_query_history": "No query history yet",
        "load_history": "Load",
        "clear_history": "Clear my history",
        "account_management": "User management",
        "create_user": "Create user",
        "user_id": "User ID",
        "display_name": "Display name",
        "password": "Password",
        "role": "Role",
        "active": "Active",
        "save_user": "Save user",
        "reset_password": "Reset password",
        "user_created": "User created",
        "user_saved": "User updated",
        "password_reset": "Password updated",
        "answer_modes": {
            "explanation": "Explanation",
            "derivation": "Derivation",
            "implementation": "Implementation",
            "literature_review": "Literature Review",
            "code_generation": "Code Generation",
        },
        "job_created": "Job status: {job_id} ({status})",
        "job_failed": "Job failed: {message}",
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
        "initialization_failed": "Knowledge base unavailable: {error} Ask an administrator to upload and index a PDF.",
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


def rel_path(path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def paper_label(path, manifest: dict[str, dict]) -> str:
    item = manifest.get(path.name, {})
    title = item.get("title") or path.stem.replace("-", " ")
    topic = item.get("topic")
    source = "Seed" if PAPERS_LIBRARY_DIR in path.parents else "Upload"
    return f"[{source}] {title}" + (f" · {topic}" if topic else "")


def render_account_gate(user_store: LocalUserStore) -> UserAccount:
    """Require a local account before exposing the research workspace."""
    current_user_id = st.session_state.get("current_user_id")
    if current_user_id:
        current_user = user_store.get_user(current_user_id)
        if current_user and current_user.active:
            return current_user
        st.session_state.pop("current_user_id", None)
        st.session_state.pop("messages", None)

    st.title("⚡ FluxMind")
    if not user_store.has_users():
        st.subheader("创建管理员 / Create admin")
        st.caption("首次启动只需创建一个本地管理员账户。")
        with st.form("initial_admin_form"):
            user_id = st.text_input("用户 ID / User ID", value="admin")
            display_name = st.text_input("显示名称 / Display name", value="Admin")
            password = st.text_input("密码 / Password", type="password")
            submitted = st.form_submit_button("创建并登录 / Create and sign in")
        if submitted:
            try:
                user = user_store.create_initial_admin(
                    user_id=user_id,
                    display_name=display_name,
                    password=password,
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                st.session_state["current_user_id"] = user.user_id
                st.rerun()
        st.stop()

    st.subheader("登录 / Sign in")
    with st.form("login_form"):
        user_id = st.text_input("用户 ID / User ID")
        password = st.text_input("密码 / Password", type="password")
        submitted = st.form_submit_button("登录 / Sign in")
    if submitted:
        user = user_store.authenticate(user_id, password)
        if user is None:
            st.error("用户 ID 或密码错误 / Invalid user ID or password")
        else:
            st.session_state["current_user_id"] = user.user_id
            st.session_state.pop("messages", None)
            st.rerun()
    st.stop()


def render_query_history(
    user_store: LocalUserStore,
    current_user: UserAccount,
    labels: dict,
) -> None:
    with st.expander(f"🕘 {labels['query_history']}"):
        history = user_store.list_history(current_user.user_id, limit=20)
        if not history:
            st.caption(labels["no_query_history"])
            return
        for entry in history:
            st.caption(f"{entry.created_at[:16].replace('T', ' ')} · {entry.answer_mode}")
            st.markdown(f"**{entry.question}**")
            if st.button(
                labels["load_history"],
                key=f"load_history_{entry.history_id}",
                use_container_width=True,
            ):
                st.session_state["messages"] = [
                    {"role": "user", "content": entry.question},
                    {"role": "assistant", "content": entry.answer},
                ]
                st.rerun()
        if st.button(labels["clear_history"], key="clear_query_history", use_container_width=True):
            user_store.clear_history(current_user.user_id)
            st.session_state["messages"] = []
            st.rerun()


def render_user_management(user_store: LocalUserStore, labels: dict) -> None:
    with st.expander(f"👥 {labels['account_management']}"):
        st.caption(labels["create_user"])
        with st.form("create_user_form"):
            user_id = st.text_input(labels["user_id"], key="create_user_id")
            display_name = st.text_input(labels["display_name"], key="create_display_name")
            password = st.text_input(labels["password"], type="password", key="create_password")
            role = st.selectbox(labels["role"], options=["student", "admin"], key="create_role")
            submitted = st.form_submit_button(labels["create_user"])
        if submitted:
            try:
                user_store.create_user(
                    user_id=user_id,
                    display_name=display_name,
                    password=password,
                    role=role,
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                st.success(labels["user_created"])
                st.rerun()

        users = user_store.list_users(include_inactive=True)
        selected_id = st.selectbox(
            labels["user_id"],
            options=[user.user_id for user in users],
            format_func=lambda value: next(
                f"{user.display_name} · {user.role}"
                for user in users
                if user.user_id == value
            ),
            key="manage_user_id",
        )
        selected_user = next(user for user in users if user.user_id == selected_id)
        with st.form(f"edit_user_form_{selected_id}"):
            display_name = st.text_input(
                labels["display_name"],
                value=selected_user.display_name,
                key=f"edit_display_name_{selected_id}",
            )
            role = st.selectbox(
                labels["role"],
                options=["student", "admin"],
                index=0 if selected_user.role == "student" else 1,
                key=f"edit_role_{selected_id}",
            )
            active = st.checkbox(
                labels["active"],
                value=selected_user.active,
                key=f"edit_active_{selected_id}",
            )
            save_submitted = st.form_submit_button(labels["save_user"])
        if save_submitted:
            try:
                user_store.update_user(
                    selected_id,
                    display_name=display_name,
                    role=role,
                    active=active,
                )
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                st.success(labels["user_saved"])
                st.rerun()

        with st.form(f"reset_password_form_{selected_id}"):
            new_password = st.text_input(
                labels["password"],
                type="password",
                key=f"reset_password_{selected_id}",
            )
            reset_submitted = st.form_submit_button(labels["reset_password"])
        if reset_submitted:
            try:
                user_store.set_password(selected_id, new_password)
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                st.success(labels["password_reset"])


def render_streaming_response(prompt: str, *, answer_mode: str) -> tuple[str, bool]:
    """Render a streaming answer through a stable markdown placeholder."""
    request_id = new_request_id()
    logger.info(
        "streamlit.query.start request_id=%s mode=%s chars=%s",
        request_id,
        answer_mode,
        len(prompt),
    )
    chunks: list[str] = []
    placeholder = st.empty()
    try:
        for piece in query_stream(prompt, answer_mode=answer_mode):
            chunks.append(piece)
            placeholder.markdown("".join(chunks))
    except Exception as exc:
        error = normalize_exception(exc)
        logger.exception("streamlit.query.error request_id=%s code=%s", request_id, error.code)
        error_message = streamlit_status_message(error.message, fallback=error.code)
        message = f"{error_message}\n\nRequest ID: `{request_id}`"
        placeholder.error(message)
        return message, False
    response = "".join(chunks)
    logger.info("streamlit.query.ok request_id=%s chars=%s", request_id, len(response))
    return response, True


def render_job_result(job) -> None:
    """Render a compact job outcome in the sidebar."""
    if job.status in {"queued", "running", "succeeded"}:
        st.success(text["job_created"].format(job_id=job.job_id, status=job.status))
    else:
        error_message = (job.error or {}).get("message") if isinstance(job.error, dict) else None
        message = streamlit_status_message(error_message, fallback=job.status)
        st.error(text["job_failed"].format(message=message))


def job_sidebar_summary(job) -> dict:
    """Return the useful parts of a job for the Streamlit latest-jobs panel."""
    summary = job_view(job)
    summary["artifacts"] = [job_artifact_to_dict(job, artifact) for artifact in job.artifacts]
    return summary


def render_latest_jobs(current_user: UserAccount, *, is_admin: bool) -> None:
    job_query = st.text_input(text["job_search"], value="", key="job_search")
    col_status, col_kind = st.columns(2)
    with col_status:
        job_status = st.selectbox(
            text["job_status_filter"],
            options=["", "queued", "running", "succeeded", "failed", "cancelled"],
            format_func=lambda value: value or "all",
            key="job_status_filter",
        )
    with col_kind:
        job_kind = st.selectbox(
            text["job_kind_filter"],
            options=["", "image_generation", "code_execution", "index_rebuild"],
            format_func=lambda value: value or "all",
            key="job_kind_filter",
        )
    jobs = LocalJobStore().list_latest(
        limit=5,
        status=job_status or None,
        kind=job_kind or None,
        owner_id=None if is_admin else current_user.user_id,
        q=job_query or None,
    )
    if not jobs:
        st.caption(text["no_jobs"])
        return
    for job in jobs:
        label = f"{job.status} · {job.kind} · {job.job_id}"
        with st.expander(label):
            st.json(job_sidebar_summary(job))
            if job.status in {"queued", "running"}:
                if st.button(
                    text["cancel_job"],
                    key=f"cancel_{job.job_id}",
                    use_container_width=True,
                ):
                    cancelled = get_async_job_manager().cancel(job.job_id)
                    render_job_result(cancelled or job)
                    st.rerun()
            if job.status in {"failed", "cancelled"}:
                if st.button(
                    text["retry_job"],
                    key=f"retry_{job.job_id}",
                    use_container_width=True,
                ):
                    retried = LocalJobRunner().retry(job.job_id)
                    render_job_result(retried or job)
                    st.rerun()
                if st.button(
                    text["schedule_retry"],
                    key=f"retry_later_{job.job_id}",
                    use_container_width=True,
                ):
                    retried = get_async_job_manager().schedule_retry(job.job_id, delay_s=30)
                    render_job_result(retried or job)
                    st.rerun()


def render_latest_artifacts(current_user: UserAccount, *, is_admin: bool) -> None:
    artifact_query = st.text_input(text["artifact_search"], value="", key="artifact_search")
    col_kind, col_job_kind = st.columns(2)
    with col_kind:
        artifact_kind = st.selectbox(
            text["artifact_kind_filter"],
            options=["", "image", "plot", "text", "file"],
            format_func=lambda value: value or "all",
            key="artifact_kind_filter",
        )
    with col_job_kind:
        artifact_job_kind = st.selectbox(
            text["artifact_job_filter"],
            options=["", "image_generation", "code_execution", "index_rebuild"],
            format_func=lambda value: value or "all",
            key="artifact_job_kind_filter",
        )
    artifacts = LocalArtifactRegistry().list_artifacts(
        limit=5,
        kind=artifact_kind or None,
        job_kind=artifact_job_kind or None,
        owner_id=None if is_admin else current_user.user_id,
        q=artifact_query or None,
    )
    if not artifacts:
        st.caption(text["no_artifacts"])
        return
    for artifact in artifacts:
        artifact_data = artifact_to_dict(artifact)
        label = f"{artifact_data['kind']} · {artifact.artifact_id}"
        with st.expander(label):
            st.caption(str(artifact_data["job_kind"]))
            st.caption(f"{text['artifact_id']}: {artifact.artifact_id}")
            st.caption(text["artifact_metadata"])
            st.json(artifact_data)
            try:
                path = local_artifact_path(artifact.uri)
                st.download_button(
                    text["download_artifact"],
                    data=path.read_bytes(),
                    file_name=safe_artifact_download_filename(artifact, path),
                    mime=artifact.mime_type,
                    use_container_width=True,
                    key=f"download_{artifact.artifact_id}",
                )
            except (FileNotFoundError, ValueError) as exc:
                st.caption(streamlit_error_message(exc))


def render_admin_status() -> None:
    status = collect_admin_status().to_dict()
    st.caption(text["status_jobs"])
    st.json(status["jobs"])
    st.caption(text["status_artifacts"])
    st.json(status["artifacts"])
    st.caption(text["status_corpus"])
    st.json(status["corpus"])
    st.caption(text["status_provider_failures"])
    st.json(status["providers"])
    st.caption(text["status_query_usage"])
    st.json(status["activity"])
    st.caption(text["account_management"])
    st.json(status["users"])
    st.caption(text["status_storage_inventory"])
    st.json(status["storage"])
    runtime_manifest = collect_runtime_backup_manifest()
    st.caption(text["status_runtime_manifest"])
    st.json(
        {
            "mode": runtime_manifest["mode"],
            "env_file_present": runtime_manifest["env_file_present"],
            "total_files": runtime_manifest["total_files"],
            "total_bytes": runtime_manifest["total_bytes"],
        }
    )
    st.caption(text["status_runtime_dirs"])
    st.json(status["runtime_dirs"])
    st.download_button(
        text["download_status_report"],
        data=format_admin_status_report(status).encode("utf-8"),
        file_name="fluxmind-admin-status.md",
        mime="text/markdown",
        use_container_width=True,
        key="download_admin_status_report",
    )
    st.download_button(
        text["download_metrics"],
        data=format_admin_metrics(status).encode("utf-8"),
        file_name="fluxmind-admin-metrics.prom",
        mime="text/plain",
        use_container_width=True,
        key="download_admin_metrics",
    )
    st.download_button(
        text["download_runtime_manifest"],
        data=format_runtime_backup_manifest_markdown(runtime_manifest).encode("utf-8"),
        file_name="fluxmind-runtime-manifest.md",
        mime="text/markdown",
        use_container_width=True,
        key="download_runtime_manifest",
    )
    uploaded_manifest = st.file_uploader(
        text["runtime_restore_manifest_upload"],
        type=["json"],
        key="runtime_restore_manifest_upload",
    )
    if uploaded_manifest is not None:
        try:
            restore_manifest = json.loads(uploaded_manifest.getvalue().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            st.error(
                streamlit_error_text(text["runtime_restore_invalid_manifest"], exc)
            )
        else:
            restore_check = collect_runtime_restore_check(restore_manifest)
            st.caption(text["runtime_restore_check"])
            st.json(
                {
                    "ok": restore_check["ok"],
                    "manifest_errors": restore_check["manifest_errors"],
                    "checked_groups": restore_check["checked_groups"],
                    "checked_files": restore_check["checked_files"],
                    "missing_groups": restore_check["missing_groups"],
                    "mismatched_groups": restore_check["mismatched_groups"],
                    "missing_files": restore_check["missing_files"],
                    "mismatched_files": restore_check["mismatched_files"],
                }
            )
            st.download_button(
                text["download_runtime_restore_check"],
                data=format_runtime_restore_check_markdown(restore_check).encode("utf-8"),
                file_name="fluxmind-runtime-restore-dry-run.md",
                mime="text/markdown",
                use_container_width=True,
                key="download_runtime_restore_check",
            )


def render_retention_preview() -> None:
    col_upload, col_artifact, col_limit = st.columns(3)
    with col_upload:
        upload_days = st.number_input(
            text["upload_retention_days"],
            min_value=0,
            max_value=3650,
            value=30,
            step=1,
            key="retention_upload_days",
        )
    with col_artifact:
        artifact_days = st.number_input(
            text["artifact_retention_days"],
            min_value=0,
            max_value=3650,
            value=30,
            step=1,
            key="retention_artifact_days",
        )
    with col_limit:
        limit = st.number_input(
            text["retention_limit"],
            min_value=1,
            max_value=500,
            value=25,
            step=1,
            key="retention_limit",
        )

    preview = collect_retention_preview(
        upload_days=int(upload_days),
        artifact_days=int(artifact_days),
        limit=int(limit),
    )
    st.json(
        {
            "delete_enabled": preview["delete_enabled"],
            "candidate_count": preview["candidate_count"],
            "candidate_bytes": preview["candidate_bytes"],
        }
    )
    st.caption(text["retention_uploads"])
    st.json([item for item in preview["candidates"] if item["kind"] == "upload"])
    st.caption(text["retention_artifacts"])
    st.json([item for item in preview["candidates"] if item["kind"] == "artifact"])
    if preview["delete_enabled"]:
        if st.button(text["retention_delete"], key="retention_delete", use_container_width=True):
            result = apply_retention_delete(
                upload_days=int(upload_days),
                artifact_days=int(artifact_days),
                limit=int(limit),
            )
            st.caption(text["retention_delete_result"])
            st.json(
                {
                    "deleted_count": result["deleted_count"],
                    "deleted_bytes": result["deleted_bytes"],
                    "errors": result["errors"],
                }
            )


def render_runtime_events() -> None:
    event_query = st.text_input(text["event_search"], value="", key="event_search")
    col_kind, col_code = st.columns(2)
    with col_kind:
        event_kind = st.selectbox(
            text["event_kind_filter"],
            options=RUNTIME_EVENT_KIND_FILTER_OPTIONS,
            format_func=lambda value: value or "all",
            key="event_kind_filter",
        )
    with col_code:
        event_code = st.text_input(text["event_code_filter"], value="", key="event_code_filter")
    query = (event_query or "").strip()
    events = [
        runtime_event_to_dict(event, include_request_id=False)
        for event in list_runtime_events(
            kind=event_kind or None,
            code=event_code or None,
            q=query or None,
            limit=10,
        )
    ]
    if not events:
        st.caption(text["no_jobs"])
        return
    st.json(events)


user_store = LocalUserStore()
current_user = render_account_gate(user_store)
is_admin = current_user.role == "admin"


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
    st.caption(
        text["signed_in_as"].format(
            name=current_user.display_name,
            role=current_user.role,
        )
    )
    if st.button(text["logout"], key="logout", use_container_width=True):
        st.session_state.pop("current_user_id", None)
        st.session_state.pop("messages", None)
        st.rerun()
    answer_mode = st.selectbox(
        text["answer_mode"],
        options=list(text["answer_modes"]),
        format_func=lambda value: text["answer_modes"][value],
        key="answer_mode",
    )
    render_query_history(user_store, current_user, text)
    if is_admin:
        render_user_management(user_store, text)
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
            disabled=not is_admin,
        )
        if st.button(text["apply_selection"], use_container_width=True, disabled=not is_admin):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                with st.spinner(text["rebuilding"]):
                    paths = [selectable_by_rel[key] for key in selected]
                    _, chunks = rebuild_vector_store_from_pdfs(paths)
                    st.success(f"{text['rebuilt']} ({chunks} chunks)")
                    st.rerun()
        if st.button(text["save_selection"], use_container_width=True, disabled=not is_admin):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                set_active_paper_source_paths(selected)
                st.success(text["selection_saved"])
                st.rerun()
        if st.button(text["run_index_job"], use_container_width=True, disabled=not is_admin):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                with st.spinner(text["rebuilding"]):
                    job = get_async_job_manager().enqueue_index_rebuild(
                        selected,
                        ownership=account_ownership(current_user),
                    )
                    render_job_result(job)
        with st.expander(text["corpus_profiles"]):
            profile_store = CorpusProfileStore()
            profile_name = st.text_input(
                text["profile_name"],
                value="",
                key="corpus_profile_name",
                disabled=not is_admin,
            )
            if st.button(text["save_profile"], use_container_width=True, disabled=not is_admin):
                if not selected:
                    st.warning(text["select_at_least_one"])
                else:
                    profile_store.upsert_profile(
                        name=profile_name or "Active corpus",
                        source_paths=selected,
                    )
                    st.success(text["profile_saved"])
                    st.rerun()
            profiles = profile_store.list_profiles()
            if profiles:
                selected_profile = st.selectbox(
                    text["corpus_profiles"],
                    options=[profile.profile_id for profile in profiles],
                    format_func=lambda profile_id: next(
                        profile.name
                        for profile in profiles
                        if profile.profile_id == profile_id
                    ),
                    key="corpus_profile_select",
                    disabled=not is_admin,
                )
                try:
                    profile_status = collect_corpus_profile_status(selected_profile)
                    profile_report = format_corpus_profile_status_report(profile_status)
                    st.download_button(
                        text["download_profile_report"],
                        data=profile_report,
                        file_name=safe_corpus_profile_report_filename(selected_profile),
                        mime="text/markdown",
                        key="corpus_profile_report_download",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(
                        text["profile_report_failed"].format(
                            error=streamlit_error_message(exc)
                        )
                    )
                if st.button(text["activate_profile"], use_container_width=True, disabled=not is_admin):
                    profile = profile_store.get_profile(selected_profile)
                    set_active_paper_source_paths(profile.source_paths)
                    st.success(text["selection_saved"])
                    st.rerun()
                if st.button(
                    text["activate_profile_rebuild"],
                    use_container_width=True,
                    disabled=not is_admin,
                ):
                    profile = profile_store.get_profile(selected_profile)
                    set_active_paper_source_paths(profile.source_paths)
                    with st.spinner(text["rebuilding"]):
                        job = get_async_job_manager().enqueue_index_rebuild(
                            profile.source_paths,
                            ownership=account_ownership(current_user),
                        )
                        render_job_result(job)
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
        disabled=not is_admin,
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(text["indexing"].format(filename=uf.name)):
                try:
                    saved_path, n_chunks = ingest_uploaded_pdf(uf.read(), uf.name)
                    st.success(text["indexed_chunks"].format(filename=saved_path.name, chunks=n_chunks))
                except ValueError as exc:
                    st.error(streamlit_error_text(text["upload_failed"], exc))

    st.caption(f"Max upload: {MAX_UPLOAD_SIZE_MB} MB")

    st.divider()
    st.subheader(f"🧪 {text['jobs']}")
    with st.expander(text["mock_image_job"]):
        image_backend = (IMAGE_PROVIDER_BACKEND or "local-mock").strip().lower()
        openai_image_backends = {"openai", "openai-image", "openai-images", "gpt-image", "gpt-image-2"}
        image_key_missing = image_backend in openai_image_backends and not OPENAI_IMAGE_API_KEY
        st.caption(text["image_backend_caption"].format(backend=IMAGE_PROVIDER_BACKEND or "local-mock"))
        if image_backend in openai_image_backends:
            st.caption(text["image_model_caption"].format(model=OPENAI_IMAGE_MODEL))
        if image_key_missing:
            st.warning(text["image_missing_key"])
        image_template = st.selectbox(
            text["mock_image_template"],
            options=[
                "generic",
                "sliding-mode-observer",
                "pmsm-control-loop",
                "paper-figure-redraft",
            ],
            key="mock_image_template",
        )
        image_prompt = st.text_area(
            text["mock_image_prompt"],
            value="Draw a sliding-mode observer block diagram",
            key="mock_image_prompt",
            height=80,
        )
        if st.button(text["run_mock_image"], use_container_width=True, disabled=image_key_missing):
            job = get_async_job_manager().enqueue_image_generation(
                request=ImageGenerationRequest(
                    prompt=image_prompt,
                    diagram_template=image_template,
                ),
                ownership=account_ownership(current_user),
            )
            render_job_result(job)

    with st.expander(text["python_job"]):
        st.caption(text["code_execution_backend_caption"].format(backend=CODE_EXECUTION_BACKEND))
        if CODE_EXECUTION_BACKEND == "docker":
            st.caption(text["code_execution_image_caption"].format(image=DOCKER_PYTHON_EXECUTION_IMAGE))
        python_template = st.selectbox(
            text["python_template"],
            options=list(PYTHON_EXECUTION_TEMPLATES),
            format_func=lambda value: text["execution_templates"].get(
                value,
                value.replace("_", " ").title(),
            ),
            key="python_execution_template",
        )
        entrypoint = st.text_input(
            text["python_entrypoint"],
            value="main.py",
            key="python_entrypoint",
        )
        code = st.text_area(
            text["python_files"],
            value=PYTHON_EXECUTION_TEMPLATES[python_template],
            key=f"python_job_code_{python_template}",
            height=220,
        )
        if st.button(text["run_python_job"], use_container_width=True):
            job = get_async_job_manager().enqueue_local_python(
                CodeExecutionRequest(
                    language="python",
                    entrypoint=entrypoint,
                    files={entrypoint: code},
                    timeout_s=10,
                ),
                ownership=account_ownership(current_user),
            )
            render_job_result(job)

    with st.expander(text["octave_job"]):
        st.caption(text["code_execution_backend_caption"].format(backend=CODE_EXECUTION_BACKEND))
        if CODE_EXECUTION_BACKEND == "docker":
            st.caption(text["code_execution_image_caption"].format(image=DOCKER_OCTAVE_EXECUTION_IMAGE))
        octave_template = st.selectbox(
            text["octave_template"],
            options=list(OCTAVE_EXECUTION_TEMPLATES),
            format_func=lambda value: text["execution_templates"].get(
                value,
                value.replace("_", " ").title(),
            ),
            key="octave_execution_template",
        )
        octave_entrypoint = st.text_input(
            text["octave_entrypoint"],
            value="main.m",
            key="octave_entrypoint",
        )
        octave_code = st.text_area(
            text["octave_files"],
            value=OCTAVE_EXECUTION_TEMPLATES[octave_template],
            key=f"octave_job_code_{octave_template}",
            height=180,
        )
        if st.button(text["run_octave_job"], use_container_width=True):
            job = get_async_job_manager().enqueue_local_octave(
                CodeExecutionRequest(
                    language="octave",
                    entrypoint=octave_entrypoint,
                    files={octave_entrypoint: octave_code},
                    timeout_s=10,
                ),
                ownership=account_ownership(current_user),
            )
            render_job_result(job)

    with st.expander(text["latest_jobs"]):
        render_latest_jobs(current_user, is_admin=is_admin)

    with st.expander(text["latest_artifacts"]):
        render_latest_artifacts(current_user, is_admin=is_admin)

    if is_admin:
        with st.expander(text["admin_status"]):
            if st.button(text["refresh_status"], use_container_width=True):
                st.rerun()
            render_admin_status()
            st.divider()
            st.caption(text["retention_preview"])
            render_retention_preview()
            st.divider()
            st.caption(text["runtime_events"])
            render_runtime_events()

    st.divider()
    st.subheader(f"ℹ️ {text['about']}")
    st.markdown(text["about_text"])

    # Decorative footer egg — presenter-only demo walkthrough.
    if st.button("🥚", key="bg_easter", help=None):
        show_demo_guide()

# ── Init vector store on first run ──
if "store_initialized" not in st.session_state:
    with st.spinner(text["initializing"]):
        try:
            build_vector_store()
        except Exception as exc:
            st.error(streamlit_error_text(text["initialization_failed"], exc))
            st.stop()
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
        response, succeeded = render_streaming_response(
            prompt,
            answer_mode=st.session_state.answer_mode,
        )
    st.session_state.messages.append({"role": "assistant", "content": response})
    if succeeded:
        try:
            user_store.record_query(
                user_id=current_user.user_id,
                question=prompt,
                answer=response,
                answer_mode=st.session_state.answer_mode,
            )
        except (OSError, sqlite3.Error) as exc:
            st.warning(str(exc))
