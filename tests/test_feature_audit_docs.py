import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATTERN = re.compile(r"^(GET|POST|PUT|DELETE)\s+(/\S+)", re.MULTILINE)


def fastapi_routes() -> set[tuple[str, str]]:
    tree = ast.parse((PROJECT_ROOT / "api.py").read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if not isinstance(decorator.func.value, ast.Name):
                continue
            if decorator.func.value.id != "app":
                continue
            if decorator.func.attr.upper() not in {"GET", "POST", "PUT", "DELETE"}:
                continue
            if not decorator.args:
                continue
            route_arg = decorator.args[0]
            if isinstance(route_arg, ast.Constant) and isinstance(route_arg.value, str):
                routes.add((decorator.func.attr.upper(), route_arg.value))
    return routes


def documented_routes() -> set[tuple[str, str]]:
    text = (PROJECT_ROOT / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")
    return set(ROUTE_PATTERN.findall(text))


def test_feature_audit_documents_all_fastapi_routes():
    missing = fastapi_routes() - documented_routes()

    assert missing == set()


def test_feature_audit_records_current_evaluation_boundaries():
    text = (PROJECT_ROOT / "docs" / "FEATURE_AUDIT.md").read_text(encoding="utf-8")

    assert "Product platform layer        incomplete" in text
    assert "No-key execution providers    verified" in text
    assert "Feature audit drift" in text
    assert "Provider activation, MATLAB licensing, identity, quotas, and billing" in text


def test_docs_index_points_to_feature_audit():
    text = (PROJECT_ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    assert "docs/FEATURE_AUDIT.md" in text
    assert "Feature coverage audit" in text
