from __future__ import annotations

import numpy as np
import pytest

from hailmary.clustering.visualization import (
    ClusterTrajectory,
    clip_track_at_time,
    cluster_choices,
    runway_choices,
    select_trajectories,
)
from hailmary.data import RawADSBTrack


def _trajectory(runway: str, cluster: str, flight_id: str) -> ClusterTrajectory:
    return ClusterTrajectory(
        runway=runway,
        cluster=cluster,
        flight_id=flight_id,
        callsign=flight_id,
        lat_deg=np.array([32.0, 32.1]),
        lon_deg=np.array([-97.1, -97.0]),
    )


def test_choices_and_filters_cover_all_runway_and_runway_cluster_views() -> None:
    trajectories = (
        _trajectory("RW18R", "10", "A"),
        _trajectory("RW18R", "2", "B"),
        _trajectory("RW17C", "alpha", "C"),
    )

    assert runway_choices(trajectories) == ("RW17C", "RW18R")
    assert cluster_choices(trajectories, "RW18R") == ("2", "10")
    assert select_trajectories(trajectories) == trajectories
    assert [
        item.flight_id for item in select_trajectories(trajectories, runway="RW18R")
    ] == ["A", "B"]
    assert [
        item.flight_id
        for item in select_trajectories(trajectories, runway="RW18R", cluster="2")
    ] == ["B"]
    with pytest.raises(ValueError, match="together with a runway"):
        select_trajectories(trajectories, cluster="2")


def test_clip_track_interpolates_terminal_entry_and_keeps_later_samples() -> None:
    track = RawADSBTrack(
        flight_id="TEST",
        callsign="TEST",
        icao24="abc123",
        time_s=np.array([0.0, 10.0, 20.0]),
        lat_deg=np.array([32.0, 33.0, 34.0]),
        lon_deg=np.array([-99.0, -98.0, -97.0]),
        heading_deg=np.array([90.0, 90.0, 90.0]),
        geoaltitude_m=np.array([1000.0, 500.0, 0.0]),
    )

    lat, lon = clip_track_at_time(track, 5.0)

    np.testing.assert_allclose(lat, [32.5, 33.0, 34.0])
    np.testing.assert_allclose(lon, [-98.5, -98.0, -97.0])


def test_visualization_help_is_headless(capsys) -> None:
    from hailmary.clustering.visualization import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "hailmary-visualize-clusters" in capsys.readouterr().out
