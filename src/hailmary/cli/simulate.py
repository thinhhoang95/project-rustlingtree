"""Run a deterministic scenario definition assembled from JSON and NPZ variants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from hailmary.adapters import HailmaryScheduleView
from hailmary.clustering import canonical_json_dumps
from hailmary.scenario import (
    ActionStationDefinition,
    FlightDefinition,
    MaterializedExogenousEvent,
    ResourceCrossingDefinition,
    ResourceDefinition,
    ScenarioDefinition,
)
from hailmary.simulator import EventBatchResult, Simulator

from .build_templates import read_variant_npz


def _load_variant(record: object, *, root: Path) -> object:
    if isinstance(record, str):
        path = Path(record)
    elif isinstance(record, Mapping) and "npz" in record:
        path = Path(str(record["npz"]))
    else:
        raise ValueError("each scenario variant must be an NPZ path or an object containing 'npz'")
    if not path.is_absolute():
        path = root / path
    return read_variant_npz(path)


def _resource_crossings(
    payload: Mapping[str, Any],
    *,
    variant: object,
) -> tuple[ResourceCrossingDefinition, ...]:
    records = payload.get("resource_crossings")
    if records is not None:
        return tuple(ResourceCrossingDefinition(**item) for item in records)
    return tuple(
        ResourceCrossingDefinition(
            resource_id=str(getattr(item, "resource_id")),
            s_m=float(getattr(item, "s_m")),
            station_index=index,
        )
        for index, item in enumerate(getattr(variant, "resource_crossings", ()))
    )


def load_scenario(path: str | Path) -> ScenarioDefinition:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scenario JSON must contain an object")
    variants = tuple(_load_variant(item, root=source.parent) for item in payload.get("variants", ()))
    if not variants:
        raise ValueError("scenario JSON must reference at least one trajectory variant")
    variant_by_id = {str(getattr(item, "variant_id")): item for item in variants}
    resources = tuple(ResourceDefinition(**item) for item in payload.get("resources", ()))
    flights: list[FlightDefinition] = []
    for raw_flight in payload.get("flights", ()):
        record = dict(raw_flight)
        baseline_variant_id = str(record["baseline_variant_id"])
        try:
            variant = variant_by_id[baseline_variant_id]
        except KeyError as exc:
            raise ValueError(f"flight references unknown variant {baseline_variant_id!r}") from exc
        flights.append(
            FlightDefinition(
                flight_id=str(record["flight_id"]),
                release_time_s=float(record["release_time_s"]),
                baseline_variant_id=baseline_variant_id,
                cluster_id=str(record.get("cluster_id", getattr(variant, "cluster_id", ""))),
                callsign=str(record.get("callsign", "")),
                icao24=str(record.get("icao24", "")),
                runway=str(record.get("runway", "")),
                observed_release_time_s=(
                    None
                    if record.get("observed_release_time_s") is None
                    else float(record["observed_release_time_s"])
                ),
                release_offset_s=float(record.get("release_offset_s", 0.0)),
                action_stations=tuple(
                    ActionStationDefinition(**item) for item in record.get("action_stations", ())
                ),
                resource_crossings=_resource_crossings(record, variant=variant),
                metadata=record.get("metadata", {}),
            )
        )
    if not flights:
        raise ValueError("scenario JSON must contain at least one flight")
    exogenous = tuple(
        MaterializedExogenousEvent(**item) for item in payload.get("exogenous_events", ())
    )
    return ScenarioDefinition(
        scenario_id=str(payload["scenario_id"]),
        seed=int(payload["seed"]),
        flights=tuple(sorted(flights, key=lambda item: item.flight_id)),
        resources=tuple(sorted(resources, key=lambda item: item.resource_id)),
        variants=variants,
        exogenous_events=exogenous,
        decision_trigger_kinds=tuple(
            payload.get(
                "decision_trigger_kinds",
                (
                    "EXOGENOUS_DISTURBANCE",
                    "FLIGHT_RELEASED",
                    "ACTION_STATION_CROSSED",
                    "RESOURCE_CROSSED",
                ),
            )
        ),
        weather=payload.get("weather"),
        metadata=payload.get("metadata", {}),
    )


def _batch_payload(batch: EventBatchResult) -> dict[str, Any]:
    return {
        "time_s": batch.time_s,
        "events": [event.to_dict() for event in batch.events],
        "decision_epoch": None if batch.decision_epoch is None else asdict(batch.decision_epoch),
        "state_id_before": batch.state_id_before,
        "state_id_after": batch.state_id_after,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-simulate",
        description="Run a deterministic event-driven Hailmary scenario from JSON/NPZ artifacts.",
    )
    parser.add_argument("--scenario", type=Path, required=True, help="scenario-definition JSON")
    parser.add_argument("--output", type=Path, required=True, help="simulation trace JSON")
    parser.add_argument("--horizon", type=float, default=None, help="optional absolute scenario time")
    parser.add_argument("--no-schedule", action="store_true", help="omit the compatibility arrival schedule")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    definition = load_scenario(args.scenario)
    simulator = Simulator(definition)
    if args.horizon is None:
        batches = simulator.run()
    else:
        batches = simulator.advance_until(args.horizon)
    trace: dict[str, Any] = {
        "schema_version": "hailmary.simulation-trace.v1",
        "scenario_id": definition.scenario_id,
        "definition_hash": definition.definition_hash,
        "batches": [_batch_payload(item) for item in batches],
        "final_state": simulator.snapshot(),
    }
    if not args.no_schedule:
        trace["arrival_schedule"] = HailmaryScheduleView(simulator).arrival_schedule()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json_dumps(trace) + "\n", encoding="utf-8")
    print(
        canonical_json_dumps(
            {
                "batch_count": len(batches),
                "dynamic_content_hash": simulator.dynamic_content_hash,
                "output": args.output.resolve().as_posix(),
                "scenario_id": definition.scenario_id,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "load_scenario", "main"]
