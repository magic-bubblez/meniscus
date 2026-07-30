from __future__ import annotations

import ast
from pathlib import Path
import tomllib


def test_mcp_recall_is_the_single_read_tool():
    source_path = Path(__file__).parents[1] / "mcp_server.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "meniscus_recall"
    )

    assert function.args.args[0].arg == "query"
    assert "MCP_INSTRUCTIONS" in source
    assert "def meniscus_query(" not in source
    assert "def meniscus_recent(" not in source


def test_mcp_dependency_stays_on_supported_major():
    project_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text())

    assert project["project"]["optional-dependencies"]["mcp"] == [
        "mcp[cli]>=1.27,<2"
    ]
