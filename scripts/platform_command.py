from __future__ import annotations

import shlex
import sys
from pathlib import Path


def is_windows() -> bool:
    return sys.platform.startswith("win")


def python_script_command(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *args]


def display_command(argv: list[str]) -> str:
    return " ".join(_quote(str(arg)) for arg in argv)


def _quote(value: str) -> str:
    if is_windows():
        if not value or any(char.isspace() for char in value) or any(char in value for char in '"&|<>^'):
            return '"' + value.replace('"', '\\"') + '"'
        return value
    return shlex.quote(value)
