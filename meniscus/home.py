"""Resolve and initialize the Meniscus data directory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DEFAULT_HOME = "~/.meniscus"
ENV_FILENAME = ".env"
CONFIG_FILENAME = "config.toml"
DATABASE_FILENAME = "meniscus.db"

MENISCUS_HOME: Path = Path(
    os.environ.get("MENISCUS_HOME", DEFAULT_HOME)
).expanduser().resolve()


def ensure_home() -> Path:
    MENISCUS_HOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    MENISCUS_HOME.chmod(0o700)
    return MENISCUS_HOME


def write_private_text(path: Path, content: str) -> None:
    ensure_home()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temp_path = Path(temporary.name)
        temp_path.chmod(0o600)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_env() -> None:
    from dotenv import load_dotenv

    home_env = MENISCUS_HOME / ENV_FILENAME
    if home_env.exists():
        load_dotenv(home_env)
