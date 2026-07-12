"""Canonical serialization and stable identifiers.

The simulator deliberately separates reproducible content hashes from lineage
identities.  The functions here contain no Python object identities and reject
non-finite numbers, which makes them safe for artifacts and branch audits.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import enum
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def canonical_data(value: Any) -> Any:
    """Convert supported domain values to deterministic JSON-compatible data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical data cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return canonical_data(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("canonical data cannot contain NaN or infinity")
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": canonical_data(value.tolist()),
        }
    if isinstance(value, enum.Enum):
        return canonical_data(value.value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        pairs = sorted(((str(key), item) for key, item in value.items()), key=lambda pair: pair[0])
        return {key: canonical_data(item) for key, item in pairs}
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [canonical_data(item) for item in value]
        return sorted(converted, key=lambda item: canonical_json(item))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return canonical_data(to_dict())
    raise TypeError(f"unsupported canonical value: {type(value).__qualname__}")


def canonical_json(value: Any) -> str:
    """Serialize with sorted keys and a platform-independent compact form."""

    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any, *, namespace: str = "hailmary") -> str:
    payload = f"{namespace}\0{canonical_json(value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 20) -> str:
    if not prefix or any(character.isspace() for character in prefix):
        raise ValueError("prefix must be a non-empty token")
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")
    return f"{prefix}_{content_hash(value, namespace=prefix)[:length]}"


def provenance_state_id(
    dynamic_content_hash: str,
    *,
    parent_state_id: str | None,
    branch_label: str,
    lineage_sequence: int = 0,
) -> str:
    return stable_id(
        "state",
        {
            "dynamic_content_hash": dynamic_content_hash,
            "parent_state_id": parent_state_id,
            "branch_label": branch_label,
            "lineage_sequence": int(lineage_sequence),
        },
        length=32,
    )
