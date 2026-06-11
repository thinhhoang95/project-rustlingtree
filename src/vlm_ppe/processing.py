from __future__ import annotations

import pandas as pd

from vlm_ppe.geo.resample import arc_length_resample_points


def resample_track_frame(tracks: pd.DataFrame, n_resample: int) -> pd.DataFrame:
    rows: list[dict] = []
    for flight_id, group in tracks.groupby("flight_id", sort=False):
        ordered = group.sort_values(["time", "seq"], kind="stable")
        points = ordered[["x_nm", "y_nm"]].to_numpy(dtype=float)
        resampled, s_nm = arc_length_resample_points(points, n_resample)
        first = ordered.iloc[0]
        for station_index, ((x_nm, y_nm), s_value) in enumerate(zip(resampled, s_nm, strict=True)):
            rows.append(
                {
                    "flight_id": str(flight_id),
                    "callsign": str(first["callsign"]),
                    "icao24": str(first["icao24"]),
                    "operation": str(first["operation"]),
                    "runway": str(first["runway"]),
                    "station_index": int(station_index),
                    "s_fraction": float(station_index / (n_resample - 1)),
                    "s_nm": float(s_value),
                    "track_length_nm": float(s_nm[-1]),
                    "x_nm": float(x_nm),
                    "y_nm": float(y_nm),
                }
            )
    if not rows:
        raise ValueError("no tracks were resampled")
    return pd.DataFrame(rows)
