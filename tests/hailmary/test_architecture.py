from __future__ import annotations

import ast
from pathlib import Path


def test_hailmary_core_does_not_import_workflow_or_mutable_manager_packages() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "hailmary"
    forbidden = {"mcp_tools", "vlm_ppe", "fastapi", "langgraph"}
    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        if path.name.startswith("._"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.relative_to(package_root)}:{node.lineno}: {name}")
    assert violations == []
