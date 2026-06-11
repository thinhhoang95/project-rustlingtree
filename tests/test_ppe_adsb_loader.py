from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlm_ppe.config import load_config
from vlm_ppe.geo.projection import LocalProjection
from vlm_ppe.io.adsb_loader import ingest_adsb_tracks


def test_ingest_clips_tracks_to_configured_radius(tmp_path: Path) -> None:
    center_lat = 32.0
    center_lon = -97.0
    projection = LocalProjection.from_origin(center_lat, center_lon)
    y_nm = np.asarray([90.0, 30.0, 0.0])
    lat, lon = projection.unproject_nm(np.zeros_like(y_nm), y_nm)

    catalog_path = tmp_path / "catalog.csv"
    compressed_path = tmp_path / "compressed.jsonl"
    manifest_path = tmp_path / "manifest.json"
    config_path = tmp_path / "config.yaml"

    pd.DataFrame(
        {
            "flight_id": ["T1"],
            "operation": ["arrival"],
            "runway": ["18R"],
            "event_time": [120],
            "event_lat": [center_lat],
            "event_lon": [center_lon],
            "threshold_lat": [center_lat],
            "threshold_lon": [center_lon],
        }
    ).to_csv(catalog_path, index=False)

    payload = {
        "flight_id": "T1",
        "callsign": "T1",
        "icao24": "abc123",
        "columns": ["time", "lat", "lon", "geoaltitude_m", "breakpoint_mask"],
        "points": [
            [0, float(lat[0]), float(lon[0]), 3000.0, 3],
            [60, float(lat[1]), float(lon[1]), 2000.0, 3],
            [120, float(lat[2]), float(lon[2]), 1000.0, 3],
        ],
    }
    compressed_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "test": {
                    "landings_and_departures": catalog_path.as_posix(),
                    "adsb_compressed_trajectories": compressed_path.as_posix(),
                    "default": True,
                }
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                'dataset_id: "test"',
                'operation: "arrival"',
                f'manifest_path: "{manifest_path.as_posix()}"',
                f'output_root: "{(tmp_path / "out").as_posix()}"',
                f"track_filter_center_lat: {center_lat}",
                f"track_filter_center_lon: {center_lon}",
                "track_filter_radius_nm: 60.0",
            ]
        ),
        encoding="utf-8",
    )

    tracks, track_index, coordinate_system = ingest_adsb_tracks(load_config(config_path))

    distance_nm = np.hypot(tracks["x_nm"].to_numpy(dtype=float), tracks["y_nm"].to_numpy(dtype=float))
    assert float(distance_nm.max()) <= 60.0 + 1e-6
    assert int(track_index.loc[0, "point_count"]) == 3
    assert float(track_index.loc[0, "track_length_nm"]) == pytest.approx(60.0)
    assert int(tracks.iloc[0]["time"]) == 30
    assert float(tracks.iloc[0]["x_nm"]) == pytest.approx(0.0, abs=1e-6)
    assert float(tracks.iloc[0]["y_nm"]) == pytest.approx(60.0, abs=1e-6)
    assert coordinate_system.origin_lat == pytest.approx(center_lat)
    assert coordinate_system.origin_lon == pytest.approx(center_lon)
