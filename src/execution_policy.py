"""Request-level safety policy for no-key code execution providers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from src.capabilities import CodeExecutionRequest


DEFAULT_POLICY_PROFILE = "local-safe-v1"
DISABLED_POLICY = "disabled"
POLICY_VIOLATION_EXIT_CODE = 126

BLOCKED_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("package_install_command", r"\b(pip|conda|apt-get|apt|apk|yum|npm|curl|wget)\b"),
    ("shell_escape", r"(^|\n)\s*!"),
)
OCTAVE_BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("octave_shell_call", r"\b(system|unix|dos)\s*\("),
    ("octave_network_call", r"\b(webread|webwrite|urlread|urlwrite|ftp)\s*\("),
    ("octave_package_install", r"\bpkg\s+install\b"),
    ("absolute_path_literal", r"['\"]\/"),
)
PYTHON_BLOCKED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
}
PYTHON_BLOCKED_ATTR_CALLS = {
    "os.popen",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.system",
    "pathlib.Path.home",
    "pathlib.Path.unlink",
    "pathlib.Path.rmdir",
    "shutil.rmtree",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
    "subprocess.run",
}


@dataclass(frozen=True)
class ExecutionPolicyViolation:
    file: str
    code: str
    message: str


@dataclass(frozen=True)
class ExecutionPolicyResult:
    profile: str
    enabled: bool
    checked_files: int
    allowed_python_imports: tuple[str, ...]
    violations: tuple[ExecutionPolicyViolation, ...]

    @property
    def allowed(self) -> bool:
        return not self.enabled or not self.violations

    def metadata(self) -> dict[str, str]:
        return {
            "execution_policy": self.profile,
            "execution_policy_enforced": "true" if self.enabled else "false",
            "execution_policy_checked_files": str(self.checked_files),
            "execution_policy_violations": str(len(self.violations)),
            "execution_policy_allowed_imports": ",".join(self.allowed_python_imports),
            "policy_violation": "false" if self.allowed else "true",
        }

    def message(self) -> str:
        if self.allowed:
            return "Execution policy passed."
        return "; ".join(
            f"{violation.file}: {violation.code}: {violation.message}"
            for violation in self.violations
        )


def parse_allowed_imports(value: str) -> tuple[str, ...]:
    imports = sorted({item.strip() for item in value.split(",") if item.strip()})
    return tuple(imports)


def evaluate_execution_policy(
    request: CodeExecutionRequest,
    *,
    profile: str,
    allowed_python_imports: str,
) -> ExecutionPolicyResult:
    """Return a no-secret policy decision before materializing or running code."""
    profile = (profile or DEFAULT_POLICY_PROFILE).strip().lower()
    allowed_imports = parse_allowed_imports(allowed_python_imports)
    if profile == DISABLED_POLICY:
        return ExecutionPolicyResult(
            profile=profile,
            enabled=False,
            checked_files=0,
            allowed_python_imports=allowed_imports,
            violations=(),
        )

    violations: list[ExecutionPolicyViolation] = []
    checked = 0
    for name, content in request.files.items():
        if request.language == "python" and name.endswith(".py"):
            checked += 1
            violations.extend(_evaluate_python_file(name, content, allowed_imports))
        elif request.language in {"octave", "matlab"} and name.lower().endswith((".m", ".octave")):
            checked += 1
            violations.extend(_evaluate_octave_file(name, content))
        else:
            violations.extend(_evaluate_common_text(name, content))

    return ExecutionPolicyResult(
        profile=profile,
        enabled=True,
        checked_files=checked,
        allowed_python_imports=allowed_imports,
        violations=tuple(violations),
    )


def _evaluate_python_file(
    name: str,
    content: str,
    allowed_imports: tuple[str, ...],
) -> list[ExecutionPolicyViolation]:
    violations = _evaluate_common_text(name, content)
    try:
        tree = ast.parse(content, filename=name)
    except SyntaxError:
        return violations

    allowed = set(allowed_imports)
    import_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                import_aliases[alias.asname or root] = root
                if root not in allowed:
                    violations.append(
                        ExecutionPolicyViolation(
                            file=name,
                            code="python_import_not_allowed",
                            message=f"import `{root}` is not allowed by the execution policy",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root:
                for alias in node.names:
                    import_aliases[alias.asname or alias.name] = root
                if root not in allowed:
                    violations.append(
                        ExecutionPolicyViolation(
                            file=name,
                            code="python_import_not_allowed",
                            message=f"import `{root}` is not allowed by the execution policy",
                        )
                    )
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func, import_aliases)
            if call_name in PYTHON_BLOCKED_CALLS or call_name in PYTHON_BLOCKED_ATTR_CALLS:
                violations.append(
                    ExecutionPolicyViolation(
                        file=name,
                        code="python_call_not_allowed",
                        message=f"call `{call_name}` is not allowed by the execution policy",
                    )
                )
            if call_name in {"open", "pathlib.Path", "Path"} and _first_arg_is_absolute_path(node):
                violations.append(
                    ExecutionPolicyViolation(
                        file=name,
                        code="absolute_path_not_allowed",
                        message=f"call `{call_name}` uses an absolute path literal",
                    )
                )
    return violations


def _evaluate_octave_file(name: str, content: str) -> list[ExecutionPolicyViolation]:
    violations = _evaluate_common_text(name, content)
    for code, pattern in OCTAVE_BLOCKED_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE):
            violations.append(
                ExecutionPolicyViolation(
                    file=name,
                    code=code,
                    message=f"pattern `{code}` is not allowed by the execution policy",
                )
            )
    return violations


def _evaluate_common_text(name: str, content: str) -> list[ExecutionPolicyViolation]:
    violations: list[ExecutionPolicyViolation] = []
    for code, pattern in BLOCKED_TEXT_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE | re.MULTILINE):
            violations.append(
                ExecutionPolicyViolation(
                    file=name,
                    code=code,
                    message=f"pattern `{code}` is not allowed by the execution policy",
                )
            )
    return violations


def _call_name(node: ast.AST, import_aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value, import_aliases)
        root = parent.split(".", 1)[0]
        mapped = import_aliases.get(root, root)
        suffix = parent.split(".", 1)[1:] if "." in parent else []
        parts = [mapped, *suffix, node.attr]
        return ".".join(item for item in parts if item)
    return ""


def _first_arg_is_absolute_path(node: ast.Call) -> bool:
    if not node.args:
        return False
    first = node.args[0]
    return isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("/")
