"""Typed reader for the repository data manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetResources:
    """Resolved resource paths for one immutable dataset entry."""

    dataset_id: str
    manifest_path: Path
    resources: tuple[tuple[str, Path], ...]
    is_default: bool = False

    def __post_init__(self) -> None:
        dataset_id = str(self.dataset_id).strip()
        if not dataset_id:
            raise ValueError("dataset_id must be nonempty")
        normalized = tuple(sorted(((str(key), Path(value)) for key, value in self.resources), key=lambda item: item[0]))
        if any(not key for key, _ in normalized):
            raise ValueError("resource names must be nonempty")
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("resource names must be unique")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        object.__setattr__(self, "resources", normalized)

    def get(self, name: str, default: Path | None = None) -> Path | None:
        return dict(self.resources).get(name, default)

    def require(self, name: str) -> Path:
        value = self.get(name)
        if value is None:
            raise KeyError(f"dataset {self.dataset_id!r} has no {name!r} resource")
        return value

    @property
    def landings_and_departures(self) -> Path | None:
        return self.get("landings_and_departures")

    @property
    def raw_adsb(self) -> Path | None:
        return self.get("raw_adsb") or self.get("adsb_raw")

    @property
    def compressed_adsb(self) -> Path | None:
        return self.get("adsb_compressed_trajectories")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "default": bool(self.is_default),
            "resources": {key: value.as_posix() for key, value in self.resources},
        }


@dataclass(frozen=True)
class DataManifest:
    path: Path
    datasets: tuple[DatasetResources, ...]

    def __post_init__(self) -> None:
        datasets = tuple(sorted(self.datasets, key=lambda item: item.dataset_id))
        if not datasets:
            raise ValueError("data manifest contains no datasets")
        if len({item.dataset_id for item in datasets}) != len(datasets):
            raise ValueError("data manifest contains duplicate dataset IDs")
        if sum(item.is_default for item in datasets) > 1:
            raise ValueError("data manifest contains more than one default dataset")
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(self, "datasets", datasets)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(item.dataset_id for item in self.datasets)

    def select(self, dataset_id: str | None = None) -> DatasetResources:
        if dataset_id is not None:
            requested = str(dataset_id).strip()
            for item in self.datasets:
                if item.dataset_id == requested:
                    return item
            raise KeyError(f"unknown dataset_id {requested!r}; available: {list(self.dataset_ids)}")
        defaults = [item for item in self.datasets if item.is_default]
        if len(defaults) == 1:
            return defaults[0]
        if len(self.datasets) == 1:
            return self.datasets[0]
        raise ValueError("dataset_id is required because the manifest has no default")

    def to_dict(self) -> dict[str, Any]:
        return {item.dataset_id: item.to_dict() for item in self.datasets}


def _resolve_resource_path(manifest_path: Path, value: object, *, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest resource {key!r} must be a nonempty path string")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_manifest(path: str | Path) -> DataManifest:
    """Load and resolve a repository-style ``data_manifest.json``."""

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("data manifest must be a nonempty JSON object")

    datasets: list[DatasetResources] = []
    for raw_dataset_id, raw_entry in sorted(payload.items(), key=lambda item: str(item[0])):
        dataset_id = str(raw_dataset_id).strip()
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"manifest entry {dataset_id!r} must be a JSON object")
        default_value = raw_entry.get("default", False)
        if not isinstance(default_value, bool):
            raise ValueError(f"manifest entry {dataset_id!r} default must be boolean")
        resources = tuple(
            (str(key), _resolve_resource_path(manifest_path, value, key=str(key)))
            for key, value in raw_entry.items()
            if key != "default" and value is not None
        )
        datasets.append(
            DatasetResources(
                dataset_id=dataset_id,
                manifest_path=manifest_path,
                resources=resources,
                is_default=default_value,
            )
        )
    return DataManifest(path=manifest_path, datasets=tuple(datasets))


def resolve_dataset_resources(path: str | Path, dataset_id: str | None = None) -> DatasetResources:
    return load_manifest(path).select(dataset_id)


load_data_manifest = load_manifest

