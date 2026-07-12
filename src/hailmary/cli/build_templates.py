"""Compile medoid tracks into immutable trajectory-template NPZ artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from hailmary.adapters import SIMAPAdapter
from hailmary.clustering import ClusterLibrary, canonical_json_dumps
from hailmary.templates import MedoidTrack, TemplateCompiler
from hailmary.templates.models import (
    ActionProvenance,
    ResourceCrossing,
    TrajectoryVariant,
    VariantDiagnostics,
)


_VARIANT_ARRAY_NAMES = (
    "s_m",
    "lat_deg",
    "lon_deg",
    "east_m",
    "north_m",
    "altitude_m",
    "cas_mps",
    "tas_mps",
    "ground_speed_mps",
    "command_cas_mps",
    "reference_command_cas_mps",
    "lower_cas_mps",
    "upper_cas_mps",
    "elapsed_time_s",
)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write NPZ members in stable order with a fixed ZIP timestamp."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _load_medoid_tracks(path: Path) -> dict[str, MedoidTrack]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            required = {"track_ids", "time_s", "lat_deg", "lon_deg", "altitude_m"}
            missing = sorted(required.difference(payload.files))
            if missing:
                raise ValueError(f"medoid NPZ is missing fields: {missing}")
            ids = [str(item) for item in np.asarray(payload["track_ids"]).tolist()]
            values = {name: np.asarray(payload[name], dtype=np.float64) for name in required - {"track_ids"}}
            ground_speed = (
                None
                if "ground_speed_mps" not in payload.files
                else np.asarray(payload["ground_speed_mps"], dtype=np.float64)
            )
        if any(array.ndim != 2 or array.shape[0] != len(ids) for array in values.values()):
            raise ValueError("medoid NPZ profile arrays must have shape (track, sample)")
        if ground_speed is not None and (
            ground_speed.ndim != 2 or ground_speed.shape[0] != len(ids)
        ):
            raise ValueError("medoid NPZ ground_speed_mps must have shape (track, sample)")
        return {
            flight_id: MedoidTrack(
                flight_id=flight_id,
                time_s=values["time_s"][index],
                lat_deg=values["lat_deg"][index],
                lon_deg=values["lon_deg"][index],
                altitude_m=values["altitude_m"][index],
                ground_speed_mps=None
                if ground_speed is None
                else ground_speed[index],
            )
            for index, flight_id in enumerate(ids)
        }
    if path.suffix.lower() != ".json":
        raise ValueError("medoid-track input must be .json or .npz")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "tracks" in payload:
        payload = payload["tracks"]
    records: dict[str, Any]
    if isinstance(payload, list):
        records = {str(item["flight_id"]): item for item in payload}
    elif isinstance(payload, dict):
        records = {str(key): value for key, value in payload.items()}
    else:
        raise ValueError("medoid JSON must be a mapping or list of track records")
    tracks: dict[str, MedoidTrack] = {}
    for flight_id, record in sorted(records.items()):
        if not isinstance(record, Mapping):
            raise ValueError(f"medoid track {flight_id!r} must be an object")
        tracks[flight_id] = MedoidTrack(
            flight_id=flight_id,
            time_s=record["time_s"],
            lat_deg=record["lat_deg"],
            lon_deg=record["lon_deg"],
            altitude_m=record["altitude_m"],
            ground_speed_mps=record.get("ground_speed_mps"),
        )
    return tracks


def _diagnostics_from_payload(payload: Mapping[str, Any]) -> VariantDiagnostics:
    values = dict(payload)
    values["details"] = tuple((str(item[0]), item[1]) for item in values.get("details", ()))
    return VariantDiagnostics(**values)


def _provenance_from_payload(payload: Mapping[str, Any]) -> ActionProvenance:
    values = dict(payload)
    values["realization_metadata"] = tuple(
        (str(item[0]), item[1]) for item in values.get("realization_metadata", ())
    )
    return ActionProvenance(**values)


def write_variant_npz(variant: TrajectoryVariant, path: str | Path) -> Path:
    output = Path(path)
    metadata = {
        "schema_version": variant.schema_version,
        "template_id": variant.template_id,
        "cluster_id": variant.cluster_id,
        "variant_id": variant.variant_id,
        "resource_crossings": [asdict(item) for item in variant.resource_crossings],
        "diagnostics": asdict(variant.diagnostics),
        "action_provenance": asdict(variant.action_provenance),
    }
    arrays = {name: np.asarray(getattr(variant, name), dtype=np.float64) for name in _VARIANT_ARRAY_NAMES}
    arrays["metadata_json"] = np.asarray(canonical_json_dumps(metadata))
    _write_deterministic_npz(output, arrays)
    return output


def read_variant_npz(path: str | Path) -> TrajectoryVariant:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        missing = sorted({*_VARIANT_ARRAY_NAMES, "metadata_json"}.difference(payload.files))
        if missing:
            raise ValueError(f"trajectory NPZ is missing fields: {missing}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        arrays = {name: np.asarray(payload[name], dtype=np.float64) for name in _VARIANT_ARRAY_NAMES}
    variant = TrajectoryVariant(
        template_id=str(metadata["template_id"]),
        cluster_id=str(metadata["cluster_id"]),
        resource_crossings=tuple(ResourceCrossing(**item) for item in metadata["resource_crossings"]),
        diagnostics=_diagnostics_from_payload(metadata["diagnostics"]),
        action_provenance=_provenance_from_payload(metadata["action_provenance"]),
        schema_version=str(metadata["schema_version"]),
        variant_id=str(metadata["variant_id"]),
        **arrays,
    )
    return variant


def _action_station_payload(station: object) -> dict[str, Any]:
    return {
        "entry_order": int(getattr(station, "entry_order")),
        "kind": str(getattr(station, "kind")),
        "grid_index": int(getattr(station, "grid_index")),
        "s_m": float(getattr(station, "s_m")),
        "east_m": float(getattr(station, "east_m")),
        "north_m": float(getattr(station, "north_m")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-build-templates",
        description="Compile cluster medoid raw tracks into deterministic template NPZ artifacts.",
    )
    parser.add_argument("--clusters", type=Path, required=True, help="cluster-library JSON")
    parser.add_argument("--medoid-tracks", type=Path, required=True, help="raw medoid JSON/NPZ")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--station-count", type=int, default=512)
    parser.add_argument("--payload-kg", type=float, default=12_000.0)
    parser.add_argument(
        "--no-simap-validation",
        action="store_true",
        help="compile with A320 envelopes but skip the public SIMAP ReferencePath validation pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    library = ClusterLibrary.read(args.clusters)
    tracks = _load_medoid_tracks(args.medoid_tracks)
    adapter = SIMAPAdapter(payload_kg=args.payload_kg)
    compiler = TemplateCompiler(
        station_count=args.station_count,
        aircraft_config=adapter.resolved_aircraft_config,
        validator=None if args.no_simap_validation else adapter,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for medoid in library.medoids:
        try:
            track = tracks[medoid.medoid_flight_id]
        except KeyError as exc:
            raise ValueError(f"missing raw medoid track {medoid.medoid_flight_id!r}") from exc
        template = compiler.compile(
            track,
            cluster_id=str(medoid.cluster_id),
            member_count=medoid.member_count,
            dataset_id=library.dataset_id,
            airport_id=library.airport,
            runway_id=library.runway,
            dispersion_m=medoid.mean_distance_m,
            threshold_resource_id=f"{library.airport}:{library.runway}:threshold",
        )
        filename = f"{template.template_id}.npz"
        write_variant_npz(template.baseline_variant, output_dir / filename)
        records.append(
            {
                "cluster_id": template.cluster_id,
                "medoid_flight_id": template.medoid_flight_id,
                "member_count": template.member_count,
                "dispersion_m": template.dispersion_m,
                "template_id": template.template_id,
                "variant_id": template.baseline_variant.variant_id,
                "variant_npz": filename,
                "speed_action_stations": [
                    _action_station_payload(item) for item in template.speed_action_stations
                ],
                "path_stretch_stations": [
                    _action_station_payload(item) for item in template.path_stretch_stations
                ],
                "diagnostics": asdict(template.baseline_variant.diagnostics),
                "provenance": dict(template.provenance),
            }
        )
    manifest = {
        "schema_version": "hailmary.template-library.v1",
        "cluster_artifact_content_hash": library.artifact_content_hash,
        "dataset_id": library.dataset_id,
        "airport": library.airport,
        "runway": library.runway,
        "aircraft_assumption": {
            "typecode": adapter.resolved_aircraft_config.typecode,
            "engine_name": adapter.resolved_aircraft_config.engine_name,
            "mass_kg": adapter.resolved_aircraft_config.mass_kg,
            "payload_kg": args.payload_kg,
        },
        "templates": sorted(records, key=lambda item: str(item["cluster_id"])),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")
    print(
        canonical_json_dumps(
            {
                "manifest": manifest_path.as_posix(),
                "template_count": len(records),
                "cluster_artifact_content_hash": library.artifact_content_hash,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "read_variant_npz", "write_variant_npz"]
