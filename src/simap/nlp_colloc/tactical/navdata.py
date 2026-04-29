from __future__ import annotations

import csv
from pathlib import Path

from .models import AltitudeConstraint, PathWaypoint


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def load_fix_catalog(path: str | Path) -> dict[str, PathWaypoint]:
    catalog: dict[str, PathWaypoint] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = (row.get("identifier") or "").strip().upper()
            if not identifier:
                continue
            if not (row.get("latitude_deg") or "").strip() or not (row.get("longitude_deg") or "").strip():
                continue
            catalog[identifier] = PathWaypoint(
                identifier=identifier,
                lat_deg=float(row["latitude_deg"]),
                lon_deg=float(row["longitude_deg"]),
                source=(row.get("fix_type") or "fix").strip() or "fix",
                elevation_ft=_optional_float(row.get("elevation_ft")),
            )
    return catalog


def parse_altitude_ft(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip().upper()
    if not text:
        return None
    if text.startswith("FL"):
        return float(text[2:]) * 100.0
    return float(text)


def load_procedure_altitude_constraints(
    path: str | Path,
    *,
    procedure_identifier: str | None = None,
    transition_identifier: str | None = None,
) -> tuple[AltitudeConstraint, ...]:
    procedure = None if procedure_identifier is None else procedure_identifier.upper()
    transition = None if transition_identifier is None else transition_identifier.upper()
    constraints: list[AltitudeConstraint] = []

    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if procedure is not None and (row.get("sid_star_approach_identifier") or "").upper() != procedure:
                continue
            if transition is not None and (row.get("transition_identifier") or "").upper() != transition:
                continue

            fix = (row.get("fix_identifier") or "").strip().upper()
            if not fix:
                continue
            altitude_1 = parse_altitude_ft(row.get("altitude"))
            altitude_2 = parse_altitude_ft(row.get("altitude_2"))
            if altitude_1 is None and altitude_2 is None:
                continue
            if altitude_1 is not None and altitude_2 is not None:
                lower_ft = min(altitude_1, altitude_2)
                upper_ft = max(altitude_1, altitude_2)
            else:
                lower_ft = altitude_1 if altitude_1 is not None else altitude_2
                upper_ft = altitude_1 if altitude_1 is not None else altitude_2
            constraints.append(AltitudeConstraint(fix_identifier=fix, lower_ft=lower_ft, upper_ft=upper_ft))

    return tuple(constraints)
