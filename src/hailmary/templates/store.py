"""Thread-safe template/variant registry with deterministic cache keys."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
from pathlib import Path
from threading import RLock

from hailmary.errors import ArtifactValidationError
from hailmary.ids import stable_id
from hailmary.templates.models import ClusterTemplate, TrajectoryVariant


class TemplateStore:
    """Own immutable arrays once and share them across all scenario forks."""

    def __init__(self, templates: Iterable[ClusterTemplate] = ()) -> None:
        self._lock = RLock()
        self._templates: dict[str, ClusterTemplate] = {}
        self._variants: dict[str, TrajectoryVariant] = {}
        self._cache: dict[str, str] = {}
        for template in templates:
            self.add_template(template)

    def add_template(self, template: ClusterTemplate) -> None:
        with self._lock:
            existing = self._templates.get(template.template_id)
            if existing is not None and existing.content_hash != template.content_hash:
                raise ArtifactValidationError(f"template ID collision: {template.template_id}")
            self._templates[template.template_id] = template
            self._variants[template.baseline_variant.variant_id] = template.baseline_variant

    def add_variant(self, variant: TrajectoryVariant, *, cache_key: str | None = None) -> TrajectoryVariant:
        with self._lock:
            existing = self._variants.get(variant.variant_id)
            if existing is not None and existing.content_hash != variant.content_hash:
                raise ArtifactValidationError(f"variant ID collision: {variant.variant_id}")
            self._variants[variant.variant_id] = variant
            if cache_key is not None:
                self._cache[cache_key] = variant.variant_id
            return variant

    def template(self, template_id: str) -> ClusterTemplate:
        with self._lock:
            return self._templates[template_id]

    def variant(self, variant_id: str) -> TrajectoryVariant:
        with self._lock:
            return self._variants[variant_id]

    def template_for_cluster(self, cluster_id: str) -> ClusterTemplate:
        with self._lock:
            matches = [item for item in self._templates.values() if item.cluster_id == cluster_id]
        if len(matches) != 1:
            raise KeyError(f"expected one template for cluster {cluster_id}; found {len(matches)}")
        return matches[0]

    def cache_key(self, namespace: str, *parts: object) -> str:
        return stable_id("variant-cache", {"namespace": namespace, "parts": parts}, length=40)

    def get_or_compile(
        self,
        cache_key: str,
        compiler: Callable[[], TrajectoryVariant],
    ) -> TrajectoryVariant:
        with self._lock:
            variant_id = self._cache.get(cache_key)
            if variant_id is not None:
                return self._variants[variant_id]
        compiled = compiler()
        with self._lock:
            variant_id = self._cache.get(cache_key)
            if variant_id is not None:
                return self._variants[variant_id]
            return self.add_variant(compiled, cache_key=cache_key)

    @property
    def templates(self) -> tuple[ClusterTemplate, ...]:
        with self._lock:
            return tuple(sorted(self._templates.values(), key=lambda item: item.template_id))

    @property
    def variants(self) -> tuple[TrajectoryVariant, ...]:
        with self._lock:
            return tuple(sorted(self._variants.values(), key=lambda item: item.variant_id))

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            baseline_ids = {
                template.baseline_variant.variant_id for template in self._templates.values()
            }
            return {
                "schema_version": "hailmary.template_store.v1",
                "templates": [item.to_dict() for item in self.templates],
                "additional_variants": [
                    item.to_dict()
                    for item in self.variants
                    if item.variant_id not in baseline_ids
                ],
                "cache": dict(sorted(self._cache.items())),
            }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        if output.suffix.lower() != ".json":
            output = output / "hailmary_templates.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TemplateStore":
        if str(payload.get("schema_version", "")) != "hailmary.template_store.v1":
            raise ArtifactValidationError("unsupported template-store schema")
        raw_templates = payload.get("templates", ())
        if not isinstance(raw_templates, (tuple, list)):
            raise ArtifactValidationError("template-store templates must be a list")
        store = cls(ClusterTemplate.from_dict(item) for item in raw_templates)  # type: ignore[arg-type]
        raw_variants = payload.get("additional_variants", ())
        if not isinstance(raw_variants, (tuple, list)):
            raise ArtifactValidationError("template-store variants must be a list")
        for item in raw_variants:
            store.add_variant(TrajectoryVariant.from_dict(item))  # type: ignore[arg-type]
        raw_cache = payload.get("cache", {})
        if not isinstance(raw_cache, Mapping):
            raise ArtifactValidationError("template-store cache must be a mapping")
        for key, variant_id in raw_cache.items():
            if str(variant_id) not in {item.variant_id for item in store.variants}:
                raise ArtifactValidationError(f"cache references unknown variant: {variant_id}")
            store._cache[str(key)] = str(variant_id)
        return store

    @classmethod
    def read(cls, path: str | Path) -> "TemplateStore":
        source = Path(path)
        if source.is_dir():
            source = source / "hailmary_templates.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ArtifactValidationError("template-store JSON must contain an object")
        return cls.from_dict(payload)
