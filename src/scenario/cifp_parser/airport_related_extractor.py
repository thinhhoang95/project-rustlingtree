from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from datetime import datetime
from pathlib import Path
import warnings

import arinc424
import pandas as pd

TRANSITION_COLUMNS = [
    "section_code",
    "sid_star_approach_identifier",
    "route_type",
    "transition_identifier",
    "sequence_number",
    "fix_identifier",
    "icao_code",
    "section_code_2",
    "altitude",
    "altitude_2",
]

FIX_COLUMNS = [
    "identifier",
    "fix_type",
    "source_section_code",
    "airport_icao",
    "latitude_raw",
    "longitude_raw",
    "elevation_raw",
    "latitude_deg",
    "longitude_deg",
    "elevation_ft",
]

_PROCEDURE_SECTION_CODES = {"PD", "PE", "PF"}


@dataclass(slots=True)
class AirportProcedureExtractionSummary:
    filter_hit_count: int
    processed_row_count: int
    fixes_added_count: int
    unresolved_fix_reference_count: int
    unresolved_fix_identifier_count: int
    filter_parse_failure_count: int
    deduplicated_fix_row_count: int


@dataclass(slots=True)
class LookupFailureDetail:
    token: str
    token_kind: str
    occurrence_count: int
    lookup_steps: list[str]
    failure_reason: str
    other_matches: list[str]


@dataclass(slots=True)
class _FixIndexes:
    airport_waypoints: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    airport_waypoints_by_identifier: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    enroute_waypoints_by_identifier: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    navaids_by_identifier: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    runways: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)
    runways_by_identifier: dict[str, list[dict[str, str]]] = field(default_factory=lambda: defaultdict(list))


def build_airport_procedure_dataframes(
    cifp_path: str | Path,
    airport_icao: str,
    raw_prefix_pattern: str | None = None,
    report_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, AirportProcedureExtractionSummary]:
    """Build procedure and fix dataframes for one airport from a CIFP file."""

    airport_icao = airport_icao.strip().upper()
    cifp_path = Path(cifp_path)
    pattern = raw_prefix_pattern or f"SUSAP {airport_icao}*"

    indexes = _FixIndexes()
    transition_rows: list[dict[str, object]] = []
    unresolved_tokens: list[str] = []
    filter_parse_failure_samples: list[str] = []
    filter_hit_count = 0
    filter_parse_failure_count = 0
    matched_but_unparsed = 0

    with cifp_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            raw_line = raw_line.rstrip("\r\n")
            if fnmatchcase(raw_line, pattern):
                filter_hit_count += 1
            parsed = _parse_record(raw_line)
            if parsed is None:
                if fnmatchcase(raw_line, pattern):
                    matched_but_unparsed += 1
                    filter_parse_failure_count += 1
                    if len(filter_parse_failure_samples) < 10:
                        filter_parse_failure_samples.append(raw_line)
                continue

            _index_fix_record(indexes, parsed)

            if not _is_target_procedure_record(parsed, airport_icao, pattern, raw_line):
                continue

            transition_rows.append(_transition_row(parsed))
            unresolved_tokens.extend(_procedure_reference_tokens(parsed))

    unresolved_token_counts = Counter(unresolved_tokens)
    fixes_rows, unresolved_reference_count, deduplicated_fix_row_count, failure_details = (
        _resolve_fix_rows(indexes, airport_icao, unresolved_token_counts)
    )

    if matched_but_unparsed:
        warnings.warn(
            f"Skipped {matched_but_unparsed} procedure candidate records that the ARINC-424 parser could not read.",
            RuntimeWarning,
            stacklevel=2,
        )

    unresolved = {detail.token for detail in failure_details}
    if unresolved:
        sample = ", ".join(sorted(unresolved)[:10])
        suffix = "" if len(unresolved) <= 10 else f" (+{len(unresolved) - 10} more)"
        warnings.warn(
            f"Unresolved procedure references for {airport_icao}: {sample}{suffix}",
            RuntimeWarning,
            stacklevel=2,
        )

    transitions_df = _frame_from_rows(transition_rows, TRANSITION_COLUMNS)
    fixes_df = _frame_from_rows(fixes_rows, FIX_COLUMNS)
    summary = AirportProcedureExtractionSummary(
        filter_hit_count=filter_hit_count,
        processed_row_count=len(transition_rows),
        fixes_added_count=len(fixes_rows),
        unresolved_fix_reference_count=unresolved_reference_count,
        unresolved_fix_identifier_count=len(failure_details),
        filter_parse_failure_count=filter_parse_failure_count,
        deduplicated_fix_row_count=deduplicated_fix_row_count,
    )
    if report_path is not None:
        _write_extraction_report(
            Path(report_path),
            airport_icao=airport_icao,
            cifp_path=cifp_path,
            filter_pattern=pattern,
            summary=summary,
            failure_details=failure_details,
            filter_parse_failure_samples=filter_parse_failure_samples,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )
    return transitions_df, fixes_df, summary


def _parse_record(raw_line: str) -> dict[str, str] | None:
    record = arinc424.Record()
    if not record.read(raw_line):
        return None
    return {field.name: field.value for field in record.fields}


def _is_target_procedure_record(
    parsed: dict[str, str],
    airport_icao: str,
    raw_prefix_pattern: str,
    raw_line: str,
) -> bool:
    if not fnmatchcase(raw_line, raw_prefix_pattern):
        return False
    if _clean(parsed.get("Record Type")) != "S":
        return False
    if _clean(parsed.get("Section Code")) not in _PROCEDURE_SECTION_CODES:
        return False
    return _clean(parsed.get("Airport Identifier")) == airport_icao


def _index_fix_record(indexes: _FixIndexes, parsed: dict[str, str]) -> None:
    section_code = _clean(parsed.get("Section Code"))
    if section_code == "PC":
        identifier = _clean(parsed.get("Waypoint Identifier"))
        airport_icao = _clean(parsed.get("Region Code"))
        if identifier and airport_icao:
            indexes.airport_waypoints[(airport_icao, identifier)] = parsed
            indexes.airport_waypoints_by_identifier[identifier].append(parsed)
        return

    if section_code == "EA":
        identifier = _clean(parsed.get("Waypoint Identifier"))
        if identifier:
            indexes.enroute_waypoints_by_identifier[identifier].append(parsed)
        return

    if section_code == "D":
        identifier = _clean(parsed.get("VOR Identifier")) or _clean(parsed.get("NDB Identifier"))
        if identifier:
            indexes.navaids_by_identifier[identifier].append(parsed)
        return

    if section_code == "PG":
        identifier = _clean(parsed.get("Runway Identifier"))
        airport_icao = _clean(parsed.get("Airport ICAO Identifier")) or _clean(parsed.get("Airport Identifier"))
        if identifier and airport_icao:
            indexes.runways[(airport_icao, identifier)] = parsed
            indexes.runways_by_identifier[identifier].append(parsed)


def _transition_row(parsed: dict[str, str]) -> dict[str, object]:
    return {
        "section_code": _clean(parsed.get("Section Code")),
        "sid_star_approach_identifier": _clean(parsed.get("SID/STAR/Approach Identifier")),
        "route_type": _clean(parsed.get("Route Type")),
        "transition_identifier": _clean(parsed.get("Transition Identifier")),
        "sequence_number": _clean(parsed.get("Sequence Number")),
        "fix_identifier": _clean(parsed.get("Fix Identifier")),
        "icao_code": _clean(parsed.get("ICAO Code (2)")),
        "section_code_2": _clean(parsed.get("Section Code (2)")),
        "altitude": _clean(parsed.get("Altitude")),
        "altitude_2": _clean(parsed.get("Altitude (2)")),
    }


def _procedure_reference_tokens(parsed: dict[str, str]) -> set[str]:
    tokens: set[str] = set()
    for key in ("Transition Identifier", "Fix Identifier"):
        token = _clean(parsed.get(key))
        if token:
            tokens.add(token)
    return tokens


def _resolve_fix_rows(
    indexes: _FixIndexes,
    airport_icao: str,
    tokens: Counter[str],
) -> tuple[list[dict[str, object]], int, int, list[LookupFailureDetail]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    unresolved_reference_count = 0
    deduplicated_fix_row_count = 0
    failure_details: list[LookupFailureDetail] = []

    for token, occurrence_count in tokens.items():
        if token.startswith("RW"):
            candidate = indexes.runways.get((airport_icao, token))
            if candidate is None:
                unresolved_reference_count += occurrence_count
                failure_details.append(_runway_lookup_failure_detail(token, airport_icao, occurrence_count, indexes))
                continue
            row = _build_fix_row(token, candidate, airport_icao, "runway")
        else:
            row = _resolve_named_fix(token, airport_icao, indexes)
            if row is None:
                unresolved_reference_count += occurrence_count
                failure_details.append(_named_lookup_failure_detail(token, airport_icao, occurrence_count, indexes))
                continue

        key = (
            row["identifier"],
            row["fix_type"],
            row["source_section_code"],
            row["airport_icao"],
            row["latitude_raw"],
            row["longitude_raw"],
            row["elevation_raw"],
        )
        if key in seen:
            deduplicated_fix_row_count += 1
            continue
        seen.add(key)
        rows.append(row)

    return rows, unresolved_reference_count, deduplicated_fix_row_count, failure_details


def _resolve_named_fix(
    token: str,
    airport_icao: str,
    indexes: _FixIndexes,
) -> dict[str, object] | None:
    candidate = indexes.airport_waypoints.get((airport_icao, token))
    if candidate is not None:
        return _build_fix_row(token, candidate, _clean(candidate.get("Region Code")), "waypoint")

    candidate = _first(indexes.enroute_waypoints_by_identifier.get(token))
    if candidate is not None:
        return _build_fix_row(token, candidate, None, "waypoint")

    candidate = _first(indexes.navaids_by_identifier.get(token))
    if candidate is not None:
        fix_type = "ndb" if _clean(candidate.get("NDB Identifier")) else "vor"
        return _build_fix_row(token, candidate, None, fix_type)

    candidate = _first(indexes.airport_waypoints_by_identifier.get(token))
    if candidate is not None:
        return _build_fix_row(token, candidate, _clean(candidate.get("Region Code")), "waypoint")

    return None


def _named_lookup_failure_detail(
    token: str,
    airport_icao: str,
    occurrence_count: int,
    indexes: _FixIndexes,
) -> LookupFailureDetail:
    lookup_steps = [
        f"airport waypoint lookup for {airport_icao}/{token}",
        f"enroute waypoint lookup for {token}",
        f"navaid lookup for {token}",
        f"airport waypoint lookup by identifier for {token}",
    ]
    other_matches: list[str] = []
    if token in indexes.enroute_waypoints_by_identifier:
        other_matches.append(f"enroute waypoint records exist for {token} in other contexts")
    if token in indexes.navaids_by_identifier:
        other_matches.append(f"navaid records exist for {token} in other contexts")
    airport_waypoint_hits = indexes.airport_waypoints_by_identifier.get(token, [])
    if airport_waypoint_hits:
        airports = sorted(
            {
                _clean(hit.get("Region Code")) or _clean(hit.get("Airport Identifier")) or "unknown"
                for hit in airport_waypoint_hits
            }
        )
        other_matches.append("airport waypoint records exist at: " + ", ".join(airports[:5]))

    if token == airport_icao:
        failure_reason = (
            f"identifier {token} is the airport ICAO code, but no fix record exists with that identifier"
        )
    elif other_matches:
        failure_reason = (
            f"no KDFW-local fix matched identifier {token}; the token exists elsewhere, but not in the lookup scope"
        )
    else:
        failure_reason = f"no matching fix record was found for identifier {token}"

    return LookupFailureDetail(
        token=token,
        token_kind="named",
        occurrence_count=occurrence_count,
        lookup_steps=lookup_steps,
        failure_reason=failure_reason,
        other_matches=other_matches,
    )


def _runway_lookup_failure_detail(
    token: str,
    airport_icao: str,
    occurrence_count: int,
    indexes: _FixIndexes,
) -> LookupFailureDetail:
    lookup_steps = [f"runway lookup for {airport_icao}/{token}"]
    other_matches: list[str] = []
    runway_hits = indexes.runways_by_identifier.get(token, [])
    if runway_hits:
        airports = sorted(
            {
                _clean(hit.get("Airport ICAO Identifier")) or _clean(hit.get("Airport Identifier")) or "unknown"
                for hit in runway_hits
            }
        )
        other_matches.append("runway records exist at: " + ", ".join(airports[:8]))
        failure_reason = f"no runway record matched {airport_icao}/{token}; the runway identifier exists at other airports"
    else:
        failure_reason = f"no runway record matched {airport_icao}/{token}"

    return LookupFailureDetail(
        token=token,
        token_kind="runway",
        occurrence_count=occurrence_count,
        lookup_steps=lookup_steps,
        failure_reason=failure_reason,
        other_matches=other_matches,
    )


def _build_fix_row(
    token: str,
    parsed: dict[str, str],
    airport_icao: str | None,
    fix_type: str,
) -> dict[str, object]:
    source_section_code = _clean(parsed.get("Section Code"))
    latitude_raw = _clean(
        _first_field(parsed, "Waypoint Latitude", "VOR Latitude", "NDB Latitude", "Runway Latitude")
    )
    longitude_raw = _clean(
        _first_field(parsed, "Waypoint Longitude", "VOR Longitude", "NDB Longitude", "Runway Longitude")
    )
    elevation_raw = _clean(
        _first_field(
            parsed,
            "Landing Threshold Elevation",
            "VOR Elevation",
            "NDB Elevation",
            "Waypoint Elevation",
            "Airport Elevation",
            "(LTP) Ellipsoid Height",
            "(FPAP) Ellipsoid Height",
        )
    )

    return {
        "identifier": token,
        "fix_type": fix_type,
        "source_section_code": source_section_code,
        "airport_icao": airport_icao,
        "latitude_raw": latitude_raw,
        "longitude_raw": longitude_raw,
        "elevation_raw": elevation_raw,
        "latitude_deg": _parse_latitude(latitude_raw),
        "longitude_deg": _parse_longitude(longitude_raw),
        "elevation_ft": _parse_elevation(elevation_raw),
    }


def _frame_from_rows(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first(value: list[dict[str, str]] | None) -> dict[str, str] | None:
    if not value:
        return None
    return value[0]


def _first_field(parsed: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = _clean(parsed.get(name))
        if value is not None:
            return value
    return None


def _parse_latitude(value: str | None) -> float | None:
    if not value:
        return None
    if len(value) < 9:
        return None
    hemisphere = value[0]
    degrees = value[1:3]
    minutes = value[3:5]
    seconds = value[5:7]
    hundredths = value[7:9]
    try:
        raw = int(degrees) + int(minutes) / 60 + (int(seconds) + int(hundredths) / 100) / 3600
    except ValueError:
        return None
    return -raw if hemisphere == "S" else raw


def _parse_longitude(value: str | None) -> float | None:
    if not value:
        return None
    if len(value) < 10:
        return None
    hemisphere = value[0]
    degrees = value[1:4]
    minutes = value[4:6]
    seconds = value[6:8]
    hundredths = value[8:10]
    try:
        raw = int(degrees) + int(minutes) / 60 + (int(seconds) + int(hundredths) / 100) / 3600
    except ValueError:
        return None
    return -raw if hemisphere == "W" else raw


def _parse_elevation(value: str | None) -> int | None:
    if not value:
        return None
    digits = "".join(ch for ch in value if ch.isdigit() or ch == "-")
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _write_extraction_report(
    report_path: Path,
    *,
    airport_icao: str,
    cifp_path: Path,
    filter_pattern: str,
    summary: AirportProcedureExtractionSummary,
    failure_details: list[LookupFailureDetail],
    filter_parse_failure_samples: list[str],
    generated_at: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f"Generated at: {generated_at}",
        f"Airport: {airport_icao}",
        f"CIFP file: {cifp_path}",
        f"Filter pattern: {filter_pattern}",
        "",
        "Summary",
        f"- Filter hits: {summary.filter_hit_count}",
        f"- Successfully processed rows: {summary.processed_row_count}",
        f"- Fix rows added: {summary.fixes_added_count}",
        f"- Unresolved fix references: {summary.unresolved_fix_reference_count}",
        f"- Unresolved fix identifiers: {summary.unresolved_fix_identifier_count}",
        f"- Filter parse failures: {summary.filter_parse_failure_count}",
        f"- Deduplicated fix rows skipped: {summary.deduplicated_fix_row_count}",
    ]

    if failure_details:
        lines.extend(["", "Missing fix references"])
        for detail in sorted(failure_details, key=lambda item: (-item.occurrence_count, item.token)):
            lines.append(f"- {detail.token} ({detail.token_kind}): {detail.occurrence_count}")
            lines.append(f"  - Looked up: {', '.join(detail.lookup_steps)}")
            lines.append(f"  - Why it failed: {detail.failure_reason}")
            if detail.other_matches:
                for match in detail.other_matches:
                    lines.append(f"  - Other matches: {match}")
    else:
        lines.extend(["", "Missing fix references", "- None"])

    if filter_parse_failure_samples:
        lines.extend(["", "Filtered rows that the parser could not read"])
        for raw_line in filter_parse_failure_samples:
            lines.append(f"- {raw_line}")
    else:
        lines.extend(["", "Filtered rows that the parser could not read", "- None"])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
