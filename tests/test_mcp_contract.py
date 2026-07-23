from __future__ import annotations

import ast
from pathlib import Path
import tomllib


def test_mcp_query_keeps_public_query_parameter():
    source_path = Path(__file__).parents[1] / "mcp_server.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "meniscus_query"
    )

    assert function.args.args[0].arg == "query"
    assert "MCP_INSTRUCTIONS" in source_path.read_text()


def test_mcp_dependency_stays_on_supported_major():
    project_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text())

    assert project["project"]["optional-dependencies"]["mcp"] == [
        "mcp[cli]>=1.27,<2"
    ]
