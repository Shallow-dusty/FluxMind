"""FluxMind — RAG-based Copilot for Sliding Mode Control & Flux Linkage Estimation."""

import json
import sqlite3

import streamlit as st
import streamlit.components.v1 as components

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
from src.activation_suite import collect_activation_suite, format_activation_suite_markdown
from src.collaboration_readiness import (
    collect_collaboration_readiness,
    format_collaboration_readiness_markdown,
)
from src.openapi_contract import (
    collect_openapi_contract,
    format_openapi_contract_markdown,
    format_openapi_contract_snapshot_verify_markdown,
    verify_openapi_contract_snapshot,
)
from src.quality_readiness import collect_quality_readiness, format_quality_readiness_markdown
from src.product_activation_rehearsal import (
    collect_product_activation_rehearsal,
    format_product_activation_rehearsal_markdown,
)
from src.provider_runtime_rehearsal import (
    collect_provider_runtime_rehearsal,
    format_provider_runtime_rehearsal_markdown,
)
from src.storage_migration import (
    collect_platform_migration_rehearsal,
    format_storage_migration_rehearsal_markdown,
)
from src.capabilities import CodeExecutionRequest, ImageGenerationRequest
from src.artifacts import (
    LocalArtifactRegistry,
    artifact_to_public_dict,
    job_artifact_to_public_dict,
    local_artifact_path,
    safe_artifact_download_filename,
)
from src.chain import query_stream
from src.config import (
    MAX_UPLOAD_SIZE_MB,
    PAPERS_LIBRARY_DIR,
    PROJECT_ROOT,
    STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED,
    STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED,
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
from src.jobs import LocalJobRunner, LocalJobStore, get_async_job_manager, job_search_projection
from src.metadata import CorpusProfileStore, safe_corpus_profile_report_filename
from src.product_registry import LocalProductRegistry, product_registry_backend_status
from src.runtime import (
    list_runtime_events,
    logger,
    new_request_id,
    normalize_exception,
    runtime_event_to_safe_dict,
)
from src.share_links import LocalShareLinkRegistry, share_link_registry_backend_status
from src.storage_manifest import (
    collect_runtime_backup_manifest,
    collect_runtime_restore_check,
    format_runtime_backup_manifest_markdown,
    format_runtime_restore_check_markdown,
)
from scripts._safe_cli import format_os_error, sanitize_cli_error_message

DEMO_SCRIPT_PATH = PROJECT_ROOT / "docs" / "demo-script.md"

RUNTIME_EVENT_KIND_FILTER_OPTIONS = (
    "",
    "provider_failure",
    "provider_quota_guard",
    "product_quota",
    "product_rbac",
    "product_registry_admin",
    "share_link_admin",
    "query_usage",
    "retrieval_trace",
    "code_execution",
    "api_access",
    "admin_check",
    "upload_scan",
    "retention_delete",
)


def safe_streamlit_error_message(exc: BaseException) -> str:
    if isinstance(exc, (OSError, sqlite3.Error)):
        return format_os_error(exc)
    return sanitize_cli_error_message(str(exc)) or exc.__class__.__name__


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
        "product_registry_management": "本地产品 Registry",
        "product_registry_workspace_id": "Workspace ID",
        "product_registry_workspace_label": "Workspace 名称",
        "product_registry_owner_id": "Owner user ID",
        "product_registry_owner_label": "Owner 名称",
        "product_registry_create_workspace": "创建/更新 Workspace",
        "product_registry_member_user_id": "成员 user ID",
        "product_registry_member_label": "成员名称",
        "product_registry_member_role": "成员角色",
        "product_registry_add_member": "添加/更新成员",
        "product_registry_quota_metric": "Quota 指标",
        "product_registry_quota_limit": "Quota 上限",
        "product_registry_quota_window": "Quota 窗口秒数",
        "product_registry_set_quota": "设置 Quota",
        "product_registry_permission_action": "权限动作",
        "product_registry_check_permission": "检查权限",
        "product_registry_disabled": "本地产品 registry 未启用",
        "product_registry_management_disabled": "本地产品 registry 管理面未启用",
        "share_link_registry_management": "本地 Share Link Registry",
        "share_link_workspace_id": "Workspace ID",
        "share_link_created_by_user_id": "创建者 user ID",
        "share_link_resource_kind": "资源类型",
        "share_link_resource_ref": "资源引用",
        "share_link_description": "描述",
        "share_link_expires_in_s": "有效秒数",
        "share_link_max_redemptions": "最大兑换次数",
        "share_link_create": "创建 Share Link",
        "share_link_list": "列出 Share Links",
        "share_link_include_revoked": "包含已撤销",
        "share_link_limit": "返回上限",
        "share_link_token": "Share token",
        "share_link_resolve": "解析 Share token",
        "share_link_record_redeem": "记录一次兑换",
        "share_link_id": "Share link ID",
        "share_link_revoke": "撤销 Share Link",
        "share_link_registry_disabled": "本地 share-link registry 未启用",
        "share_link_management_disabled": "本地 share-link 管理面未启用",
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
        "status_api_access": "API 访问审计",
        "status_admin_checks": "Admin 检查事件",
        "status_upload_scan": "上传扫描",
        "status_execution_policy": "执行策略",
        "status_storage": "存储就绪状态",
        "status_storage_inventory": "本地存储盘点",
        "status_storage_schemas": "本地存储模式",
        "status_platform_readiness": "平台化就绪状态",
        "status_platform_migration_rehearsal": "本地平台迁移演练",
        "run_platform_migration_rehearsal": "运行平台迁移演练",
        "download_platform_migration_rehearsal_report": "下载平台迁移演练报告",
        "status_product_readiness": "产品化就绪状态",
        "status_product_activation_rehearsal": "本地产品激活演练",
        "run_product_activation_rehearsal": "运行产品激活演练",
        "download_product_activation_rehearsal_report": "下载产品激活演练报告",
        "status_collaboration_readiness": "协作能力就绪状态",
        "run_collaboration_readiness": "运行协作就绪检查",
        "download_collaboration_readiness_report": "下载协作就绪报告",
        "status_provider_readiness": "Provider 激活就绪状态",
        "status_provider_runtime_rehearsal": "本地 Provider 运行时演练",
        "run_provider_runtime_rehearsal": "运行 Provider 运行时演练",
        "download_provider_runtime_rehearsal_report": "下载 Provider 运行时演练报告",
        "status_quality_readiness": "质量成熟度就绪状态",
        "run_quality_readiness": "运行质量就绪检查",
        "quality_readiness_live_report": "可选 quality live eval JSON 报告",
        "download_quality_readiness_report": "下载质量就绪报告",
        "quality_readiness_report_invalid": "quality live eval JSON 无法解析：{error}",
        "status_activation_suite": "本地激活套件",
        "run_activation_suite": "运行本地激活套件",
        "activation_suite_live_report": "可选 live eval JSON 报告",
        "download_activation_suite_report": "下载激活套件报告",
        "activation_suite_report_invalid": "live eval JSON 无法解析：{error}",
        "status_openapi_contract": "OpenAPI 合约检查",
        "run_openapi_contract": "运行 OpenAPI 合约检查",
        "download_openapi_contract_report": "下载 OpenAPI 合约报告",
        "openapi_contract_snapshot": "可选 OpenAPI 合约 JSON 快照",
        "run_openapi_contract_verify": "校验 OpenAPI 合约快照",
        "download_openapi_contract_verify_report": "下载 OpenAPI 合约校验报告",
        "openapi_contract_snapshot_missing": "请先上传 OpenAPI 合约 JSON 快照。",
        "openapi_contract_snapshot_invalid": "OpenAPI 合约 JSON 无法解析：{error}",
        "status_runtime_manifest": "运行时备份清单",
        "status_storage_paths": "本地存储路径",
        "status_runtime_dirs": "运行目录",
        "no_jobs": "暂无任务",
        "run_index_job": "以任务重建索引",
        "mock_image_job": "生成本地 SVG 图",
        "mock_image_prompt": "图示提示词",
        "mock_image_template": "图示模板",
        "run_mock_image": "运行图示任务",
        "python_job": "运行本地 Python",
        "python_entrypoint": "入口文件",
        "python_template": "Python 模板",
        "python_files": "文件内容",
        "run_python_job": "运行 Python 任务",
        "octave_job": "运行本地 Octave 兼容脚本",
        "octave_entrypoint": "Octave 入口文件",
        "octave_template": "Octave 模板",
        "octave_files": "Octave 文件内容",
        "run_octave_job": "运行 Octave 任务",
        "execution_templates": {
            "hello": "最小输出",
            "smc_reaching_law": "SMC 趋近律响应",
            "pmsm_current_decay": "PMSM 电流响应",
        },
        "answer_mode": "回答模式",
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
        "product_registry_management": "Local product registry",
        "product_registry_workspace_id": "Workspace ID",
        "product_registry_workspace_label": "Workspace label",
        "product_registry_owner_id": "Owner user ID",
        "product_registry_owner_label": "Owner label",
        "product_registry_create_workspace": "Create/update workspace",
        "product_registry_member_user_id": "Member user ID",
        "product_registry_member_label": "Member label",
        "product_registry_member_role": "Member role",
        "product_registry_add_member": "Add/update member",
        "product_registry_quota_metric": "Quota metric",
        "product_registry_quota_limit": "Quota limit",
        "product_registry_quota_window": "Quota window seconds",
        "product_registry_set_quota": "Set quota",
        "product_registry_permission_action": "Permission action",
        "product_registry_check_permission": "Check permission",
        "product_registry_disabled": "Local product registry is not enabled",
        "product_registry_management_disabled": "Local product registry management is not enabled",
        "share_link_registry_management": "Local share-link registry",
        "share_link_workspace_id": "Workspace ID",
        "share_link_created_by_user_id": "Creator user ID",
        "share_link_resource_kind": "Resource kind",
        "share_link_resource_ref": "Resource reference",
        "share_link_description": "Description",
        "share_link_expires_in_s": "Expires in seconds",
        "share_link_max_redemptions": "Max redemptions",
        "share_link_create": "Create share link",
        "share_link_list": "List share links",
        "share_link_include_revoked": "Include revoked",
        "share_link_limit": "Limit",
        "share_link_token": "Share token",
        "share_link_resolve": "Resolve share token",
        "share_link_record_redeem": "Record redemption",
        "share_link_id": "Share link ID",
        "share_link_revoke": "Revoke share link",
        "share_link_registry_disabled": "Local share-link registry is not enabled",
        "share_link_management_disabled": "Local share-link management is not enabled",
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
        "status_api_access": "API access audit",
        "status_admin_checks": "Admin check events",
        "status_upload_scan": "Upload scan",
        "status_execution_policy": "Execution policy",
        "status_storage": "Storage readiness",
        "status_storage_inventory": "Local storage inventory",
        "status_storage_schemas": "Local storage schemas",
        "status_platform_readiness": "Platform readiness",
        "status_platform_migration_rehearsal": "Local platform migration rehearsal",
        "run_platform_migration_rehearsal": "Run platform migration rehearsal",
        "download_platform_migration_rehearsal_report": "Download platform migration rehearsal report",
        "status_product_readiness": "Product readiness",
        "status_product_activation_rehearsal": "Local product activation rehearsal",
        "run_product_activation_rehearsal": "Run product activation rehearsal",
        "download_product_activation_rehearsal_report": "Download product activation rehearsal report",
        "status_collaboration_readiness": "Collaboration readiness",
        "run_collaboration_readiness": "Run collaboration readiness",
        "download_collaboration_readiness_report": "Download collaboration readiness report",
        "status_provider_readiness": "Provider activation readiness",
        "status_provider_runtime_rehearsal": "Local provider runtime rehearsal",
        "run_provider_runtime_rehearsal": "Run provider runtime rehearsal",
        "download_provider_runtime_rehearsal_report": "Download provider runtime rehearsal report",
        "status_quality_readiness": "Quality maturity readiness",
        "run_quality_readiness": "Run quality readiness",
        "quality_readiness_live_report": "Optional quality live eval JSON report",
        "download_quality_readiness_report": "Download quality readiness report",
        "quality_readiness_report_invalid": "Quality live eval JSON could not be parsed: {error}",
        "status_activation_suite": "Local activation suite",
        "run_activation_suite": "Run local activation suite",
        "activation_suite_live_report": "Optional live eval JSON report",
        "download_activation_suite_report": "Download activation suite report",
        "activation_suite_report_invalid": "Live eval JSON could not be parsed: {error}",
        "status_openapi_contract": "OpenAPI contract",
        "run_openapi_contract": "Run OpenAPI contract check",
        "download_openapi_contract_report": "Download OpenAPI contract report",
        "openapi_contract_snapshot": "Optional OpenAPI contract JSON snapshot",
        "run_openapi_contract_verify": "Verify OpenAPI contract snapshot",
        "download_openapi_contract_verify_report": "Download OpenAPI contract verify report",
        "openapi_contract_snapshot_missing": "Upload an OpenAPI contract JSON snapshot first.",
        "openapi_contract_snapshot_invalid": "OpenAPI contract JSON could not be parsed: {error}",
        "status_runtime_manifest": "Runtime backup manifest",
        "status_storage_paths": "Local storage paths",
        "status_runtime_dirs": "Runtime directories",
        "no_jobs": "No jobs yet",
        "run_index_job": "Rebuild Index as Job",
        "mock_image_job": "Generate Local SVG",
        "mock_image_prompt": "Diagram prompt",
        "mock_image_template": "Diagram template",
        "run_mock_image": "Run Image Job",
        "python_job": "Run Local Python",
        "python_entrypoint": "Entrypoint",
        "python_template": "Python template",
        "python_files": "File contents",
        "run_python_job": "Run Python Job",
        "octave_job": "Run Local Octave-Compatible Script",
        "octave_entrypoint": "Octave entrypoint",
        "octave_template": "Octave template",
        "octave_files": "Octave file contents",
        "run_octave_job": "Run Octave Job",
        "execution_templates": {
            "hello": "Minimal output",
            "smc_reaching_law": "SMC reaching-law response",
            "pmsm_current_decay": "PMSM current response",
        },
        "answer_mode": "Answer Mode",
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


def render_streaming_response(prompt: str, *, answer_mode: str) -> str:
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
        message = f"{error.message}\n\nRequest ID: `{request_id}`"
        placeholder.error(message)
        return message
    response = "".join(chunks)
    logger.info("streamlit.query.ok request_id=%s chars=%s", request_id, len(response))
    return response


def render_job_result(job) -> None:
    """Render a compact job outcome in the sidebar."""
    if job.status in {"queued", "running", "succeeded"}:
        st.success(text["job_created"].format(job_id=job.job_id, status=job.status))
    else:
        message = (job.error or {}).get("message", job.status)
        st.error(text["job_failed"].format(message=message))


def job_sidebar_summary(job) -> dict:
    """Return a no-secret summary for the Streamlit latest-jobs panel."""
    summary = job_search_projection(job)
    summary["artifacts"] = [
        job_artifact_to_public_dict(job, artifact)
        for artifact in job.artifacts
    ]
    return summary


def render_latest_jobs() -> None:
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


def render_latest_artifacts() -> None:
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
        q=artifact_query or None,
    )
    if not artifacts:
        st.caption(text["no_artifacts"])
        return
    for artifact in artifacts:
        public_artifact = artifact_to_public_dict(artifact)
        label = f"{public_artifact['kind']} · {artifact.artifact_id}"
        with st.expander(label):
            st.caption(str(public_artifact["job_kind"]))
            st.caption(f"{text['artifact_id']}: {artifact.artifact_id}")
            st.caption(text["artifact_metadata"])
            st.json(public_artifact["metadata"])
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
                st.caption(safe_streamlit_error_message(exc))


def render_product_registry_management() -> None:
    status = product_registry_backend_status()
    st.caption(text["product_registry_management"])
    st.json(
        {
            "backend": status.get("backend", ""),
            "available": status.get("available", False),
            "reason": status.get("reason", ""),
            "users": status.get("user_count", 0),
            "workspaces": status.get("workspace_count", 0),
            "rbac_available": status.get("rbac_available", False),
            "quota_limits": status.get("quota_limit_count", 0),
            "usage_events": status.get("usage_event_count", 0),
            "billing_accounts": status.get("billing_account_count", 0),
            "management_enabled": STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED,
            "secrets_exported": status.get("secrets_exported", False),
        }
    )
    if not status.get("available", False):
        st.caption(text["product_registry_disabled"])
        return
    if not STREAMLIT_PRODUCT_REGISTRY_MANAGEMENT_ENABLED:
        st.caption(text["product_registry_management_disabled"])
        return

    registry = LocalProductRegistry()
    try:
        workspaces = registry.list_workspace_summaries(limit=50)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(safe_streamlit_error_message(exc))
        return
    st.json({"workspaces": workspaces})
    default_workspace_id = workspaces[0]["workspace_id"] if workspaces else "local-workspace"

    with st.form("product_registry_workspace_form"):
        workspace_id = st.text_input(
            text["product_registry_workspace_id"],
            value=default_workspace_id,
            key="product_registry_workspace_id",
        )
        workspace_label = st.text_input(
            text["product_registry_workspace_label"],
            value="Local workspace",
            key="product_registry_workspace_label",
        )
        owner_id = st.text_input(
            text["product_registry_owner_id"],
            value="local-user",
            key="product_registry_owner_id",
        )
        owner_label = st.text_input(
            text["product_registry_owner_label"],
            value="Local user",
            key="product_registry_owner_label",
        )
        if st.form_submit_button(text["product_registry_create_workspace"], use_container_width=True):
            try:
                workspace = registry.create_workspace(
                    workspace_id=workspace_id,
                    label=workspace_label,
                    owner_user_id=owner_id,
                    owner_label=owner_label,
                )
                st.json(registry.workspace_detail(workspace_id=workspace.workspace_id))
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("product_registry_member_form"):
        member_workspace_id = st.text_input(
            text["product_registry_workspace_id"],
            value=default_workspace_id,
            key="product_registry_member_workspace_id",
        )
        member_user_id = st.text_input(
            text["product_registry_member_user_id"],
            value="local-member",
            key="product_registry_member_user_id",
        )
        member_label = st.text_input(
            text["product_registry_member_label"],
            value="Local member",
            key="product_registry_member_label",
        )
        member_role = st.selectbox(
            text["product_registry_member_role"],
            options=["viewer", "member", "admin", "owner"],
            index=1,
            key="product_registry_member_role",
        )
        if st.form_submit_button(text["product_registry_add_member"], use_container_width=True):
            try:
                if registry.workspace_detail(workspace_id=member_workspace_id) is None:
                    st.error(text["product_registry_disabled"])
                else:
                    registry.add_member(
                        workspace_id=member_workspace_id,
                        user_id=member_user_id,
                        label=member_label,
                        role=member_role,
                    )
                    st.json(registry.workspace_detail(workspace_id=member_workspace_id))
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("product_registry_quota_form"):
        quota_workspace_id = st.text_input(
            text["product_registry_workspace_id"],
            value=default_workspace_id,
            key="product_registry_quota_workspace_id",
        )
        quota_metric = st.text_input(
            text["product_registry_quota_metric"],
            value="requests",
            key="product_registry_quota_metric",
        )
        quota_limit = st.number_input(
            text["product_registry_quota_limit"],
            min_value=0,
            max_value=1_000_000_000,
            value=1000,
            step=1,
            key="product_registry_quota_limit",
        )
        quota_window = st.number_input(
            text["product_registry_quota_window"],
            min_value=0,
            max_value=31_536_000,
            value=86400,
            step=60,
            key="product_registry_quota_window",
        )
        if st.form_submit_button(text["product_registry_set_quota"], use_container_width=True):
            try:
                if registry.workspace_detail(workspace_id=quota_workspace_id) is None:
                    st.error(text["product_registry_disabled"])
                else:
                    registry.set_quota(
                        workspace_id=quota_workspace_id,
                        metric=quota_metric,
                        limit_value=int(quota_limit),
                        window_s=int(quota_window),
                    )
                    st.json(registry.workspace_detail(workspace_id=quota_workspace_id))
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("product_registry_permission_form"):
        permission_workspace_id = st.text_input(
            text["product_registry_workspace_id"],
            value=default_workspace_id,
            key="product_registry_permission_workspace_id",
        )
        permission_user_id = st.text_input(
            text["product_registry_member_user_id"],
            value="local-member",
            key="product_registry_permission_user_id",
        )
        permission_action = st.selectbox(
            text["product_registry_permission_action"],
            options=["query", "job_submit", "corpus_write", "admin_write"],
            key="product_registry_permission_action",
        )
        if st.form_submit_button(text["product_registry_check_permission"], use_container_width=True):
            try:
                st.json(
                    registry.permission_decision(
                        workspace_id=permission_workspace_id,
                        user_id=permission_user_id,
                        action=permission_action,
                    )
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))


def render_share_link_registry_management() -> None:
    status = share_link_registry_backend_status()
    st.caption(text["share_link_registry_management"])
    st.json(
        {
            "backend": status.get("backend", ""),
            "available": status.get("available", False),
            "reason": status.get("reason", ""),
            "active_links": status.get("active_link_count", 0),
            "revoked_links": status.get("revoked_link_count", 0),
            "expired_links": status.get("expired_link_count", 0),
            "total_links": status.get("total_link_count", 0),
            "management_enabled": STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED,
            "secrets_exported": status.get("secrets_exported", False),
            "share_tokens_exported": status.get("share_tokens_exported", False),
            "share_urls_exported": status.get("share_urls_exported", False),
        }
    )
    if not status.get("available", False):
        st.caption(text["share_link_registry_disabled"])
        return
    if not STREAMLIT_SHARE_LINK_MANAGEMENT_ENABLED:
        st.caption(text["share_link_management_disabled"])
        return

    registry = LocalShareLinkRegistry()
    try:
        links = [record.to_public_dict() for record in registry.list_links(include_revoked=True, limit=20)]
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(safe_streamlit_error_message(exc))
        return
    st.json({"share_links": links})
    default_workspace_id = "local-workspace"

    with st.form("share_link_create_form"):
        workspace_id = st.text_input(
            text["share_link_workspace_id"],
            value=default_workspace_id,
            key="share_link_create_workspace_id",
        )
        created_by_user_id = st.text_input(
            text["share_link_created_by_user_id"],
            value="local-user",
            key="share_link_created_by_user_id",
        )
        resource_kind = st.selectbox(
            text["share_link_resource_kind"],
            options=["corpus_profile", "paper", "artifact", "job", "report"],
            key="share_link_resource_kind",
        )
        resource_ref = st.text_input(
            text["share_link_resource_ref"],
            value="local-corpus-profile",
            key="share_link_resource_ref",
        )
        description = st.text_input(
            text["share_link_description"],
            value="",
            key="share_link_description",
        )
        expires_in_s = st.number_input(
            text["share_link_expires_in_s"],
            min_value=60,
            max_value=31_536_000,
            value=604800,
            step=60,
            key="share_link_expires_in_s",
        )
        max_redemptions = st.number_input(
            text["share_link_max_redemptions"],
            min_value=0,
            max_value=1_000_000,
            value=1,
            step=1,
            key="share_link_max_redemptions",
        )
        if st.form_submit_button(text["share_link_create"], use_container_width=True):
            try:
                st.json(
                    registry.create_link(
                        workspace_id=workspace_id,
                        created_by_user_id=created_by_user_id,
                        resource_kind=resource_kind,
                        resource_ref=resource_ref,
                        description=description,
                        expires_in_s=int(expires_in_s),
                        max_redemptions=int(max_redemptions),
                    )
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("share_link_list_form"):
        list_workspace_id = st.text_input(
            text["share_link_workspace_id"],
            value=default_workspace_id,
            key="share_link_list_workspace_id",
        )
        include_revoked = st.checkbox(
            text["share_link_include_revoked"],
            value=True,
            key="share_link_include_revoked",
        )
        limit = st.number_input(
            text["share_link_limit"],
            min_value=1,
            max_value=200,
            value=50,
            step=1,
            key="share_link_limit",
        )
        if st.form_submit_button(text["share_link_list"], use_container_width=True):
            try:
                st.json(
                    {
                        "share_links": [
                            record.to_public_dict()
                            for record in registry.list_links(
                                workspace_id=list_workspace_id,
                                include_revoked=include_revoked,
                                limit=int(limit),
                            )
                        ],
                        "content_exported": False,
                        "secrets_exported": False,
                        "share_tokens_exported": False,
                        "share_urls_exported": False,
                    }
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("share_link_resolve_form"):
        token = st.text_input(
            text["share_link_token"],
            value="",
            type="password",
            key="share_link_resolve_token",
        )
        record_redeem = st.checkbox(
            text["share_link_record_redeem"],
            value=False,
            key="share_link_record_redeem",
        )
        if st.form_submit_button(text["share_link_resolve"], use_container_width=True):
            try:
                st.json(
                    {
                        "resolution": registry.resolve_token(
                            token,
                            record_redeem=record_redeem,
                        )
                    }
                )
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))

    with st.form("share_link_revoke_form"):
        link_id = st.text_input(
            text["share_link_id"],
            value=links[0]["link_id"] if links else "",
            key="share_link_revoke_link_id",
        )
        if st.form_submit_button(text["share_link_revoke"], use_container_width=True):
            try:
                record = registry.revoke_link(link_id)
                if record is None:
                    st.json(
                        {
                            "ok": False,
                            "reason": "share_link_not_found",
                            "secrets_exported": False,
                            "share_tokens_exported": False,
                        }
                    )
                else:
                    st.json(
                        {
                            "ok": True,
                            "share_link": record.to_public_dict(),
                            "secrets_exported": False,
                            "share_tokens_exported": False,
                            "share_urls_exported": False,
                        }
                    )
            except (OSError, sqlite3.Error, ValueError) as exc:
                st.error(safe_streamlit_error_message(exc))


def render_admin_status() -> None:
    status = collect_admin_status().to_dict()
    jobs = status["jobs"]
    artifacts = status["artifacts"]
    storage = status["storage"]
    storage_schemas = status["storage_schemas"]
    platform_readiness = status["platform_readiness"]
    corpus = status["corpus"]
    provider_failures = status["provider_failures"]
    query_usage = status["query_usage"]
    retrieval_traces = status["retrieval_traces"]
    code_execution = status["code_execution"]
    api_access = status["api_access"]
    admin_checks = status["admin_checks"]
    upload_scans = status["upload_scans"]
    config = status["config"]
    storage_readiness = config.get("storage_readiness", {})
    distributed_job_store = config.get("distributed_job_store", {})
    product_readiness = config.get("product_readiness", {})
    provider_readiness = config.get("provider_readiness", {})

    st.caption(text["status_jobs"])
    st.json(
        {
            "total": jobs["total"],
            "by_status": jobs["by_status"],
            "by_kind": jobs["by_kind"],
            "failed": jobs["failed"],
            "scheduled": jobs["scheduled"],
            "queue_health": jobs["queue_health"],
            "worker_leases": jobs["worker_leases"],
            "storage": jobs["storage"],
        }
    )
    st.caption(text["status_artifacts"])
    st.json(
        {
            "total": artifacts["total"],
            "bytes": artifacts["bytes"],
            "integrity": artifacts["integrity"],
        }
    )
    st.caption(text["status_corpus"])
    st.json(corpus)
    st.caption(text["status_provider_failures"])
    st.json(provider_failures)
    st.caption(text["status_query_usage"])
    st.json(query_usage)
    st.caption(text["status_retrieval_traces"])
    st.json(
        {
            "total_recent": retrieval_traces.get("total_recent", 0),
            "by_code": retrieval_traces.get("by_code", {}),
            "by_endpoint": retrieval_traces.get("by_endpoint", {}),
            "by_answer_mode": retrieval_traces.get("by_answer_mode", {}),
            "empty_recent": retrieval_traces.get("empty_recent", 0),
            "empty_rate": retrieval_traces.get("empty_rate", 0),
            "source_page_incomplete_recent": retrieval_traces.get("source_page_incomplete_recent", 0),
            "source_page_incomplete_rate": retrieval_traces.get("source_page_incomplete_rate", 0),
            "citation_checked_recent": retrieval_traces.get("citation_checked_recent", 0),
            "citation_failed_recent": retrieval_traces.get("citation_failed_recent", 0),
            "citation_failure_rate": retrieval_traces.get("citation_failure_rate", 0),
            "alerts": retrieval_traces.get("alerts", []),
            "alert_thresholds": retrieval_traces.get("alert_thresholds", {}),
            "context_count": retrieval_traces.get("context_count", {}),
            "duration_ms": retrieval_traces.get("duration_ms", {}),
        }
    )
    st.caption(text["status_cost_pricing"])
    st.json(query_usage.get("pricing", {}))
    st.caption(text["status_code_execution"])
    st.json(
        {
            "total_recent": code_execution.get("total_recent", 0),
            "by_code": code_execution.get("by_code", {}),
            "by_status": code_execution.get("by_status", {}),
            "failure_rate": code_execution.get("failure_rate", 0),
            "alerts": code_execution.get("alerts", []),
            "alert_thresholds": code_execution.get("alert_thresholds", {}),
            "duration_ms": code_execution.get("duration_ms", {}),
        }
    )
    st.caption(text["status_api_access"])
    st.json(
        {
            "audit_enabled": api_access.get("audit_enabled", False),
            "total_recent": api_access.get("total_recent", 0),
            "by_token_status": api_access.get("by_token_status", {}),
            "by_status_code": api_access.get("by_status_code", {}),
            "by_method": api_access.get("by_method", {}),
            "invalid_recent": api_access.get("invalid_recent", 0),
            "missing_recent": api_access.get("missing_recent", 0),
            "rate_limited_recent": api_access.get("rate_limited_recent", 0),
            "rate_limit": api_access.get("rate_limit", {}),
        }
    )
    st.caption(text["status_admin_checks"])
    st.json(
        {
            "audit_enabled": admin_checks.get("audit_enabled", False),
            "total_recent": admin_checks.get("total_recent", 0),
            "by_check": admin_checks.get("by_check", {}),
            "by_status": admin_checks.get("by_status", {}),
            "ok_recent": admin_checks.get("ok_recent", 0),
            "blocked_recent": admin_checks.get("blocked_recent", 0),
            "blocker_count_total": admin_checks.get("blocker_count_total", 0),
        }
    )
    st.caption(text["status_upload_scan"])
    st.json(
        {
            "scan_enabled": upload_scans.get("scan_enabled", False),
            "total_recent": upload_scans.get("total_recent", 0),
            "by_status": upload_scans.get("by_status", {}),
            "by_reason": upload_scans.get("by_reason", {}),
            "allowed_recent": upload_scans.get("allowed_recent", 0),
            "blocked_recent": upload_scans.get("blocked_recent", 0),
            "active_content_recent": upload_scans.get("active_content_recent", 0),
            "parse_failed_recent": upload_scans.get("parse_failed_recent", 0),
            "config": upload_scans.get("config", {}),
        }
    )
    st.caption(text["status_execution_policy"])
    st.json(
        {
            "backend": config.get("code_execution_backend", ""),
            "policy": config.get("code_execution_policy", ""),
            "allowed_imports": config.get("code_execution_allowed_imports", []),
            "output_limits": {
                "stdout_bytes": config.get("code_execution_max_stdout_bytes", 0),
                "stderr_bytes": config.get("code_execution_max_stderr_bytes", 0),
                "artifacts": config.get("code_execution_max_artifacts", 0),
                "artifact_bytes": config.get("code_execution_max_artifact_bytes", 0),
                "artifact_total_bytes": config.get("code_execution_max_artifact_total_bytes", 0),
                "artifact_candidates": config.get("code_execution_max_artifact_candidates", 0),
            },
            "docker": config.get("docker_execution", {}),
        }
    )
    st.caption(text["status_storage"])
    st.json(
        {
            "metadata": storage_readiness.get("metadata", {}),
            "object_storage": storage_readiness.get("object_storage", {}),
            "distributed_job_store": distributed_job_store,
            "external_storage_configured": storage_readiness.get("external_storage_configured", False),
            "external_storage_available": storage_readiness.get("external_storage_available", False),
        }
    )
    st.caption(text["status_storage_inventory"])
    st.json(storage)
    st.caption(text["status_storage_schemas"])
    st.json(storage_schemas)
    st.caption(text["status_platform_readiness"])
    st.json(platform_readiness)
    st.caption(text["status_platform_migration_rehearsal"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/platform-migration-rehearsal",
            "local_only": True,
            "raw_manifests_included": False,
            "checks": [
                "runtime_backup_manifest",
                "staged_restore_check",
                "storage_schema",
                "object_storage_manifest_summary",
                "job_store_manifest_summary",
            ],
        }
    )
    if st.button(
        text["run_platform_migration_rehearsal"],
        key="run_platform_migration_rehearsal",
        use_container_width=True,
    ):
        try:
            st.session_state["platform_migration_rehearsal_status"] = (
                collect_platform_migration_rehearsal()
            )
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    platform_migration_rehearsal_status = st.session_state.get(
        "platform_migration_rehearsal_status"
    )
    if platform_migration_rehearsal_status:
        st.json(platform_migration_rehearsal_status)
        st.download_button(
            text["download_platform_migration_rehearsal_report"],
            data=format_storage_migration_rehearsal_markdown(
                platform_migration_rehearsal_status
            ),
            file_name="fluxmind-platform-migration-rehearsal.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_product_readiness"])
    st.json(
        {
            "local_foundation_ready": product_readiness.get("local_foundation_ready", False),
            "activation_ready": product_readiness.get("activation_ready", False),
            "identity_quotas_billing_enabled": product_readiness.get("identity_quotas_billing_enabled", False),
            "summary": product_readiness.get("summary", {}),
            "blockers": product_readiness.get("blockers", {}),
            "advisories": product_readiness.get("advisories", []),
            "content_exported": product_readiness.get("content_exported", False),
            "secrets_exported": product_readiness.get("secrets_exported", False),
        }
    )
    st.caption(text["status_product_activation_rehearsal"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/product-activation-rehearsal",
            "local_only": True,
            "checks": [
                "api_key_lifecycle",
                "local_product_registry",
                "local_rbac",
                "local_quota",
                "billing_attribution",
            ],
        }
    )
    if st.button(
        text["run_product_activation_rehearsal"],
        key="run_product_activation_rehearsal",
        use_container_width=True,
    ):
        try:
            st.session_state["product_activation_rehearsal_status"] = (
                collect_product_activation_rehearsal()
            )
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    product_activation_rehearsal_status = st.session_state.get(
        "product_activation_rehearsal_status"
    )
    if product_activation_rehearsal_status:
        st.json(product_activation_rehearsal_status)
        st.download_button(
            text["download_product_activation_rehearsal_report"],
            data=format_product_activation_rehearsal_markdown(
                product_activation_rehearsal_status
            ),
            file_name="fluxmind-product-activation-rehearsal.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_collaboration_readiness"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/collaboration-readiness",
            "local_only": True,
            "safe_default": {
                "private_corpora_enabled": False,
                "share_links_enabled": False,
            },
            "checks": [
                "private_corpus_policy_matrix",
                "share_link_policy_matrix",
                "product_registry_prerequisite",
                "product_rbac_guard_prerequisite",
                "share_link_token_store_prerequisite",
            ],
        }
    )
    if st.button(
        text["run_collaboration_readiness"],
        key="run_collaboration_readiness",
        use_container_width=True,
    ):
        try:
            st.session_state["collaboration_readiness_status"] = (
                collect_collaboration_readiness()
            )
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    collaboration_readiness_status = st.session_state.get(
        "collaboration_readiness_status"
    )
    if collaboration_readiness_status:
        st.json(collaboration_readiness_status)
        st.download_button(
            text["download_collaboration_readiness_report"],
            data=format_collaboration_readiness_markdown(
                collaboration_readiness_status
            ),
            file_name="fluxmind-collaboration-readiness.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_provider_readiness"])
    st.json(
        {
            "local_foundation_ready": provider_readiness.get("local_foundation_ready", False),
            "activation_ready": provider_readiness.get("activation_ready", False),
            "external_providers_enabled": provider_readiness.get("external_providers_enabled", False),
            "summary": provider_readiness.get("summary", {}),
            "checks": provider_readiness.get("checks", {}),
            "blockers": provider_readiness.get("blockers", {}),
            "advisories": provider_readiness.get("advisories", []),
            "content_exported": provider_readiness.get("content_exported", False),
            "secrets_exported": provider_readiness.get("secrets_exported", False),
            "connectivity_checked": provider_readiness.get("connectivity_checked", False),
        }
    )
    st.caption(text["status_provider_runtime_rehearsal"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/provider-runtime-rehearsal",
            "local_only": True,
            "checks": [
                "mock_image_generation",
                "local_python_execution",
                "octave_runtime_branch",
                "docker_readiness",
                "provider_quota_guard",
            ],
        }
    )
    if st.button(
        text["run_provider_runtime_rehearsal"],
        key="run_provider_runtime_rehearsal",
        use_container_width=True,
    ):
        try:
            st.session_state["provider_runtime_rehearsal_status"] = (
                collect_provider_runtime_rehearsal()
            )
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    provider_runtime_rehearsal_status = st.session_state.get(
        "provider_runtime_rehearsal_status"
    )
    if provider_runtime_rehearsal_status:
        st.json(provider_runtime_rehearsal_status)
        st.download_button(
            text["download_provider_runtime_rehearsal_report"],
            data=format_provider_runtime_rehearsal_markdown(
                provider_runtime_rehearsal_status
            ),
            file_name="fluxmind-provider-runtime-rehearsal.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_quality_readiness"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/quality-readiness",
            "live_report_input": "evaluate_rag_json_report",
            "targets": ["self_use", "small_group", "community"],
            "evidence_sources": [
                "corpus_manifest",
                "eval_baseline",
                "live_eval_report",
            ],
        }
    )
    quality_readiness_live_report = st.file_uploader(
        text["quality_readiness_live_report"],
        type=["json"],
        key="quality_readiness_live_report",
    )
    if st.button(text["run_quality_readiness"], key="run_quality_readiness", use_container_width=True):
        try:
            live_reports = []
            if quality_readiness_live_report is not None:
                live_reports.append(
                    json.loads(quality_readiness_live_report.getvalue().decode("utf-8"))
                )
            st.session_state["quality_readiness_status"] = collect_quality_readiness(
                live_reports=live_reports,
            )
        except json.JSONDecodeError as exc:
            st.error(text["quality_readiness_report_invalid"].format(error=exc))
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    quality_readiness_status = st.session_state.get("quality_readiness_status")
    if quality_readiness_status:
        st.json(quality_readiness_status)
        st.download_button(
            text["download_quality_readiness_report"],
            data=format_quality_readiness_markdown(quality_readiness_status),
            file_name="fluxmind-quality-readiness.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_activation_suite"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/activation-suite",
            "live_report_input": "evaluate_rag_json_report",
            "default_gate": "local_foundation",
            "local_foundation_requires": [
                "product local foundation",
                "provider local foundation",
                "storage migration rehearsal",
                "quality local foundation",
                "OpenAPI contract",
            ],
            "full_activation_requires": [
                "product activation",
                "provider activation",
                "platform migration activation",
                "community quality readiness",
            ],
        }
    )
    activation_suite_live_report = st.file_uploader(
        text["activation_suite_live_report"],
        type=["json"],
        key="activation_suite_live_report",
    )
    if st.button(text["run_activation_suite"], key="run_activation_suite", use_container_width=True):
        try:
            live_reports = []
            if activation_suite_live_report is not None:
                live_reports.append(
                    json.loads(activation_suite_live_report.getvalue().decode("utf-8"))
                )
            import api

            st.session_state["activation_suite_status"] = collect_activation_suite(
                live_reports=live_reports,
                openapi_schema=api.app.openapi(),
            )
        except json.JSONDecodeError as exc:
            st.error(text["activation_suite_report_invalid"].format(error=exc))
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    activation_suite_status = st.session_state.get("activation_suite_status")
    if activation_suite_status:
        st.json(activation_suite_status)
        st.download_button(
            text["download_activation_suite_report"],
            data=format_activation_suite_markdown(activation_suite_status),
            file_name="fluxmind-activation-suite.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.caption(text["status_openapi_contract"])
    st.json(
        {
            "on_demand": True,
            "route": "/admin/openapi-contract",
            "verify_route": "/admin/openapi-contract/verify",
            "raw_schema_included": False,
            "checks": [
                "required_path_methods",
                "operation_summary_ids",
                "operation_responses",
                "protected_auth_headers",
                "route_group_coverage",
                "snapshot_drift",
            ],
        }
    )
    if st.button(text["run_openapi_contract"], key="run_openapi_contract", use_container_width=True):
        try:
            import api

            st.session_state["openapi_contract_status"] = collect_openapi_contract(
                api.app.openapi()
            )
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    openapi_contract_status = st.session_state.get("openapi_contract_status")
    if openapi_contract_status:
        st.json(openapi_contract_status)
        st.download_button(
            text["download_openapi_contract_report"],
            data=format_openapi_contract_markdown(openapi_contract_status),
            file_name="fluxmind-openapi-contract.md",
            mime="text/markdown",
            use_container_width=True,
        )
    openapi_contract_snapshot = st.file_uploader(
        text["openapi_contract_snapshot"],
        type=["json"],
        key="openapi_contract_snapshot",
    )
    if st.button(
        text["run_openapi_contract_verify"],
        key="run_openapi_contract_verify",
        use_container_width=True,
    ):
        try:
            if openapi_contract_snapshot is None:
                st.error(text["openapi_contract_snapshot_missing"])
            else:
                snapshot = json.loads(openapi_contract_snapshot.getvalue().decode("utf-8"))
                import api

                current = collect_openapi_contract(api.app.openapi())
                st.session_state["openapi_contract_verify_status"] = (
                    verify_openapi_contract_snapshot(current, snapshot)
                )
        except json.JSONDecodeError as exc:
            st.error(text["openapi_contract_snapshot_invalid"].format(error=exc))
        except OSError as exc:
            st.error(safe_streamlit_error_message(exc))
    openapi_contract_verify_status = st.session_state.get("openapi_contract_verify_status")
    if openapi_contract_verify_status:
        st.json(openapi_contract_verify_status)
        st.download_button(
            text["download_openapi_contract_verify_report"],
            data=format_openapi_contract_snapshot_verify_markdown(
                openapi_contract_verify_status
            ),
            file_name="fluxmind-openapi-contract-verify.md",
            mime="text/markdown",
            use_container_width=True,
        )
    runtime_manifest = collect_runtime_backup_manifest()
    st.caption(text["status_runtime_manifest"])
    st.json(
        {
            "mode": runtime_manifest["mode"],
            "content_exported": runtime_manifest["content_exported"],
            "secrets_exported": runtime_manifest["secrets_exported"],
            "env_file_present": runtime_manifest["env_file_present"],
            "env_file_content_exported": runtime_manifest["env_file_content_exported"],
            "total_files": runtime_manifest["total_files"],
            "total_bytes": runtime_manifest["total_bytes"],
        }
    )
    st.caption(text["status_storage_paths"])
    st.json(
        {
            "metadata": storage_readiness.get("local_metadata_paths", []),
            "object_storage": storage_readiness.get("local_object_paths", []),
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
            st.error(text["runtime_restore_invalid_manifest"].format(error=exc))
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
                    "content_restored": restore_check["content_restored"],
                    "delete_enabled": restore_check["delete_enabled"],
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
            "mode": preview["mode"],
            "delete_enabled": preview["delete_enabled"],
            "limit": preview["limit"],
        }
    )
    st.caption(text["retention_uploads"])
    st.json(preview["uploads"])
    st.caption(text["retention_artifacts"])
    st.json(preview["artifacts"])
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
                    "mode": result["mode"],
                    "deleted_files": result["deleted_files"],
                    "deleted_bytes": result["deleted_bytes"],
                    "failed_files": result["failed_files"],
                    "uploads": result["uploads"],
                    "artifacts": result["artifacts"],
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
    safe_query = (event_query or "").strip()
    raw_events = list_runtime_events(
        kind=event_kind or None,
        code=event_code or None,
        q=None,
        limit=1000 if safe_query else 10,
    )
    events = [
        runtime_event_to_safe_dict(event, include_request_id=False)
        for event in raw_events
    ]
    if safe_query:
        events = [
            event
            for event in events
            if safe_query.casefold()
            in json.dumps(event, ensure_ascii=False, sort_keys=True).casefold()
        ][:10]
    if not events:
        st.caption(text["no_jobs"])
        return
    st.json(events)


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
    answer_mode = st.selectbox(
        text["answer_mode"],
        options=list(text["answer_modes"]),
        format_func=lambda value: text["answer_modes"][value],
        key="answer_mode",
    )
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
        if st.button(text["save_selection"], use_container_width=True):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                set_active_paper_source_paths(selected)
                st.success(text["selection_saved"])
                st.rerun()
        if st.button(text["run_index_job"], use_container_width=True):
            if not selected:
                st.warning(text["select_at_least_one"])
            else:
                with st.spinner(text["rebuilding"]):
                    job = get_async_job_manager().enqueue_index_rebuild(selected)
                    render_job_result(job)
        with st.expander(text["corpus_profiles"]):
            profile_store = CorpusProfileStore()
            profile_name = st.text_input(
                text["profile_name"],
                value="",
                key="corpus_profile_name",
            )
            if st.button(text["save_profile"], use_container_width=True):
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
                    error = normalize_exception(exc)
                    st.warning(text["profile_report_failed"].format(error=error.message))
                if st.button(text["activate_profile"], use_container_width=True):
                    profile = profile_store.get_profile(selected_profile)
                    set_active_paper_source_paths(profile.source_paths)
                    st.success(text["selection_saved"])
                    st.rerun()
                if st.button(text["activate_profile_rebuild"], use_container_width=True):
                    profile = profile_store.get_profile(selected_profile)
                    set_active_paper_source_paths(profile.source_paths)
                    with st.spinner(text["rebuilding"]):
                        job = get_async_job_manager().enqueue_index_rebuild(profile.source_paths)
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
    st.subheader(f"🧪 {text['jobs']}")
    with st.expander(text["mock_image_job"]):
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
        if st.button(text["run_mock_image"], use_container_width=True):
            job = get_async_job_manager().enqueue_mock_image(
                request=ImageGenerationRequest(
                    prompt=image_prompt,
                    diagram_template=image_template,
                )
            )
            render_job_result(job)

    with st.expander(text["python_job"]):
        python_template = st.selectbox(
            text["python_template"],
            options=list(PYTHON_EXECUTION_TEMPLATES),
            format_func=lambda value: text["execution_templates"][value],
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
                )
            )
            render_job_result(job)

    with st.expander(text["octave_job"]):
        octave_template = st.selectbox(
            text["octave_template"],
            options=list(OCTAVE_EXECUTION_TEMPLATES),
            format_func=lambda value: text["execution_templates"][value],
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
                )
            )
            render_job_result(job)

    with st.expander(text["latest_jobs"]):
        render_latest_jobs()

    with st.expander(text["latest_artifacts"]):
        render_latest_artifacts()

    with st.expander(text["admin_status"]):
        if st.button(text["refresh_status"], use_container_width=True):
            st.rerun()
        render_admin_status()
        st.divider()
        render_product_registry_management()
        st.divider()
        render_share_link_registry_management()
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
        response = render_streaming_response(prompt, answer_mode=st.session_state.answer_mode)
    st.session_state.messages.append({"role": "assistant", "content": response})
