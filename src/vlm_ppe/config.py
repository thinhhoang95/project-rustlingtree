from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vlm_ppe.schemas import PPEConfig


def load_config(path: str | Path) -> PPEConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    config = PPEConfig.model_validate(payload)
    return resolve_config_paths(config, config_path.parent)


def resolve_config_paths(config: PPEConfig, base_dir: Path) -> PPEConfig:
    payload: dict[str, Any] = config.model_dump()
    for key in ("manifest_path", "output_root"):
        value = Path(payload[key])
        if not value.is_absolute():
            candidate = (base_dir / value).resolve()
            if not candidate.exists() and base_dir.name == "configs":
                candidate = (base_dir.parent / value).resolve()
            value = candidate
        payload[key] = value
    return PPEConfig.model_validate(payload)
