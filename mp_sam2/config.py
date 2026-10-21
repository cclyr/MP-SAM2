"""Read the public YAML configurations without silently ignoring settings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def load_config(path: str | Path | None, allowed: set[str]) -> dict[str, Any]:
    if path is None:
        return {}
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration settings in {path}: {sorted(unknown)}")
    return config
