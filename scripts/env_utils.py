#!/usr/bin/env python3
"""Minimal secret loader for the repository's ignored `.env` file.

Only HF_TOKEN is loaded. Values are never printed, written to reports or
returned to callers. A platform-provided environment variable takes precedence.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_hf_token() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    env_path = ROOT / ".env"
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            token = value.strip()
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
                token = token[1:-1]
            if token:
                os.environ["HF_TOKEN"] = token
                return True
    return False
