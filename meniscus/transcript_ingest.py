"""Passive capture: read Claude Code and Codex session transcripts as turns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

_CLAUDE_GLOB = ".claude/projects/*/*.jsonl"
_CODEX_GLOB = ".codex/sessions/**/rollout-*.jsonl"
_SYSTEM_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<local-command",
    "<permissions",
    "<environment_context",
    "<user_instructions",
    "<system-reminder",
    "Caveat:",
)


@dataclass(frozen=True)
class Source:
    kind: str
    path: Path
    label: str


@dataclass(frozen=True)
class Turn:
    timestamp: str
    role: str
    text: str


def discover_sources(home: Path | None = None) -> list[Source]:
    home = home or Path.home()
    sources: list[Source] = []
    for path in sorted(home.glob(_CLAUDE_GLOB)):
        sources.append(Source("claude-code", path, f"claude-code/{path.stem}"))
    for path in sorted(home.glob(_CODEX_GLOB)):
        sources.append(Source("codex", path, f"codex/{path.stem}"))
    return sources


def _is_system(text: str) -> bool:
    stripped = text.lstrip()
    return not stripped or stripped.startswith(_SYSTEM_PREFIXES)


def _iso(raw: object) -> str:
    return str(raw or "").replace("Z", "+00:00")


def _parse_claude(obj: dict) -> Turn | None:
    if obj.get("isSidechain"):
        return None
    kind = obj.get("type")
    if kind not in ("user", "assistant"):
        return None
    content = (obj.get("message") or {}).get("content")
    if kind == "user" and isinstance(content, str):
        text = content.strip()
    elif kind == "assistant" and isinstance(content, list):
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        return None
    if _is_system(text):
        return None
    return Turn(_iso(obj.get("timestamp")), kind, text)


def _parse_codex(obj: dict) -> Turn | None:
    if obj.get("type") != "response_item":
        return None
    payload = obj.get("payload") or {}
    if payload.get("type") != "message":
        return None
    role = payload.get("role")
    if role not in ("user", "assistant"):
        return None
    text = "\n".join(
        block.get("text", "")
        for block in payload.get("content") or []
        if isinstance(block, dict) and block.get("type") in ("input_text", "output_text")
    ).strip()
    if _is_system(text):
        return None
    return Turn(_iso(obj.get("timestamp")), role, text)


_PARSERS: dict[str, Callable[[dict], "Turn | None"]] = {
    "claude-code": _parse_claude,
    "codex": _parse_codex,
}


def parse_turns(kind: str, lines: Iterable[str]) -> list[Turn]:
    parser = _PARSERS[kind]
    turns: list[Turn] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        turn = parser(obj)
        if turn is not None and turn.text:
            turns.append(turn)
    return turns


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Read whole lines written after ``offset``; leave a trailing partial line unread."""

    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    if not data:
        return [], offset
    parts = data.split(b"\n")
    consumed = len(data) - len(parts[-1])
    lines = [chunk.decode("utf-8", "replace") for chunk in parts[:-1]]
    return lines, offset + consumed
