from __future__ import annotations

import os
from datetime import datetime, date, time, timedelta, timezone
from typing import Any, Literal
from urllib.parse import urljoin

import requests
from zoneinfo import ZoneInfo


BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
ORIGIN_URL = "https://aeroapi.flightaware.com"


class AeroAPIError(RuntimeError):
    pass


def _iso_z(dt: datetime) -> str:
    """Return ISO8601 UTC string accepted by AeroAPI, e.g. 2024-10-05T00:00:00Z."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _next_page_url(next_link: str, current_url: str) -> str:
    """Resolve AeroAPI's links.next, which may be absolute, root-relative, or query-only."""
    if next_link.startswith("http"):
        return next_link
    if next_link.startswith("?"):
        return urljoin(current_url, next_link)
    if next_link.startswith("/aeroapi/"):
        return ORIGIN_URL + next_link
    if next_link.startswith("/"):
        return BASE_URL + next_link
    return urljoin(BASE_URL + "/", next_link)


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resp = session.get(url, params=params, timeout=30)
    if not resp.ok:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise AeroAPIError(f"AeroAPI request failed: HTTP {resp.status_code}: {detail}")
    return resp.json()


def _get_all_pages(
    session: requests.Session,
    endpoint: str,
    result_key: Literal["arrivals", "departures"],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    all_items: list[dict[str, Any]] = []

    while url:
        data = _get_json(session, url, params=params)
        all_items.extend(data.get(result_key, []))

        next_link = (data.get("links") or {}).get("next")
        url = _next_page_url(next_link, url) if next_link else None

        # Only pass query params on the first request; subsequent links already include cursor.
        params = {}

    return all_items


def _airport_timezone(session: requests.Session, airport_code: str) -> str:
    """Fetch airport timezone so dd/mm/yyyy can mean the airport's local calendar day."""
    data = _get_json(session, f"{BASE_URL}/airports/{airport_code}")
    return data.get("timezone") or "UTC"


def _daily_utc_intervals(
    day: date,
    airport_tz: str,
) -> list[tuple[datetime, datetime]]:
    """
    Convert one airport-local calendar day to UTC intervals.
    AeroAPI historical airport endpoints allow max 24h per call, so split if DST makes
    the local day longer than 24h.
    """
    tz = ZoneInfo(airport_tz)
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    intervals = []
    cursor = start_utc
    while cursor < end_utc:
        chunk_end = min(cursor + timedelta(hours=24), end_utc)
        intervals.append((cursor, chunk_end))
        cursor = chunk_end
    return intervals


def get_airport_takeoffs_and_arrivals(
    airport_icao: str,
    date_ddmmyyyy: str,
    api_key: str | None = None,
    *,
    airline: str | None = None,
    flight_type: Literal["General_Aviation", "Airline"] | None = None,
    airport_local_day: bool = True,
    allow_partial_today: bool = False,
) -> dict[str, Any]:
    """
    Return all historical takeoffs/departures and arrivals for an airport on a given date.

    Parameters
    ----------
    airport_icao:
        Airport ICAO code, e.g. "RJTT", "KJFK", "EGLL".
    date_ddmmyyyy:
        Date in dd/mm/yyyy, interpreted as the airport's local calendar day by default.
    api_key:
        FlightAware AeroAPI key. If omitted, uses environment variable AEROAPI_KEY.
    airline:
        Optional ICAO airline/operator filter, e.g. "UAL", "ANA", "BAW".
    flight_type:
        Optional "Airline" or "General_Aviation". Do not use with airline.
    airport_local_day:
        True = interpret dd/mm/yyyy in the airport's timezone.
        False = interpret dd/mm/yyyy as a UTC day.
    allow_partial_today:
        Historical endpoints cannot request beyond "now". If False, dates whose end is
        in the future raise ValueError. If True, the end is capped at now.

    Returns
    -------
    dict with:
        airport, date, timezone, start_utc, end_utc, departures, arrivals

    Notes
    -----
    Returned flight dictionaries are the raw AeroAPI BaseFlight objects, preserving
    all available fields such as ident, origin, destination, scheduled/estimated/actual
    gate and runway times, gates, terminals, aircraft_type, status, delays, etc.
    """
    api_key = api_key or os.getenv("AEROAPI_KEY")
    if not api_key:
        raise ValueError("Missing AeroAPI key. Pass api_key=... or set AEROAPI_KEY.")

    if airline and flight_type:
        raise ValueError("AeroAPI does not allow both airline and flight_type filters together.")

    airport = airport_icao.strip().upper()
    if not airport:
        raise ValueError("airport_icao cannot be empty.")

    try:
        requested_day = datetime.strptime(date_ddmmyyyy, "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError("date_ddmmyyyy must be in dd/mm/yyyy format, e.g. 05/10/2024.") from exc

    session = requests.Session()
    session.headers.update({
        "x-apikey": api_key,
        "Accept": "application/json; charset=UTF-8",
    })

    airport_tz = _airport_timezone(session, airport) if airport_local_day else "UTC"

    if airport_local_day:
        intervals = _daily_utc_intervals(requested_day, airport_tz)
    else:
        start = datetime.combine(requested_day, time.min, tzinfo=timezone.utc)
        intervals = [(start, start + timedelta(days=1))]

    now_utc = datetime.now(timezone.utc)
    if intervals[0][0] >= now_utc:
        raise ValueError("The historical airport endpoints only return data up to now; future dates are not supported.")

    if intervals[-1][1] > now_utc:
        if not allow_partial_today:
            raise ValueError(
                "The requested day has not fully completed yet. "
                "Set allow_partial_today=True to return results from start of day up to now."
            )
        intervals[-1] = (intervals[-1][0], now_utc)

    base_params: dict[str, Any] = {"max_pages": 40}
    if airline:
        base_params["airline"] = airline.upper()
    if flight_type:
        base_params["type"] = flight_type

    departures: list[dict[str, Any]] = []
    arrivals: list[dict[str, Any]] = []

    for start_utc, end_utc in intervals:
        params = {
            **base_params,
            "start": _iso_z(start_utc),
            "end": _iso_z(end_utc),
        }

        departures.extend(_get_all_pages(
            session,
            f"history/airports/{airport}/flights/departures",
            "departures",
            params,
        ))

        arrivals.extend(_get_all_pages(
            session,
            f"history/airports/{airport}/flights/arrivals",
            "arrivals",
            params,
        ))

    return {
        "airport": airport,
        "date": date_ddmmyyyy,
        "timezone": airport_tz,
        "start_utc": _iso_z(intervals[0][0]),
        "end_utc": _iso_z(intervals[-1][1]),
        "departures": departures,
        "arrivals": arrivals,
    }
    
if __name__ == '__main__':
    AEROAPI_KEY = "4gSUiZ0CTGya2iOkSu5xbv7BKWhVD7ql"
    
    result = get_airport_takeoffs_and_arrivals(
        airport_icao="KDFW",
        date_ddmmyyyy="01/04/2025",
        api_key=AEROAPI_KEY,
    )

    print(len(result["departures"]), "departures")
    print(len(result["arrivals"]), "arrivals")
    print(result["departures"][0])