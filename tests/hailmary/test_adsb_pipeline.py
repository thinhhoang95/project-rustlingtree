from __future__ import annotations

import numpy as np
import pytest

from hailmary.clustering import (
    TrackRejectionReason,
    build_cluster_library_from_adsb,
    prepare_adsb_tracks_for_clustering,
)
from hailmary.config import ClusteringConfig, HDBSCANScoreConfig, M_PER_NM
from hailmary.data import CatalogArrival, RawADSBTrack
from hailmary.geometry import LocalFrame

AIRPORT = "KDFW"
RUNWAY = "RW35C"
THRESHOLD_LAT_DEG = 32.9
THRESHOLD_LON_DEG = -97.0


def _arrival(
    flight_id: str,
    *,
    event_time_s: float,
    runway: str = RUNWAY,
) -> CatalogArrival:
    return CatalogArrival(
        event_time_s=event_time_s,
        flight_id=flight_id,
        callsign=flight_id,
        icao24=f"icao-{flight_id}",
        runway=runway,
        threshold_lat_deg=THRESHOLD_LAT_DEG,
        threshold_lon_deg=THRESHOLD_LON_DEG,
        airport=AIRPORT,
    )


def _raw_track(
    flight_id: str,
    *,
    north_nm: list[float],
    east_nm: list[float] | None = None,
    start_time_s: float = 0.0,
) -> RawADSBTrack:
    frame = LocalFrame(THRESHOLD_LAT_DEG, THRESHOLD_LON_DEG)
    east = np.zeros(len(north_nm)) if east_nm is None else np.asarray(east_nm)
    points_m = np.column_stack((east, np.asarray(north_nm))) * M_PER_NM
    lat, lon = frame.unproject_points(points_m)
    time_s = start_time_s + np.arange(len(north_nm), dtype=np.float64) * 60.0
    return RawADSBTrack(
        flight_id=flight_id,
        callsign=flight_id,
        icao24=f"icao-{flight_id}",
        time_s=time_s,
        lat_deg=lat,
        lon_deg=lon,
        heading_deg=np.full(len(north_nm), 180.0),
        geoaltitude_m=np.linspace(3_000.0, 300.0, len(north_nm)),
    )


def _pipeline_config(*, n_resample: int = 11) -> ClusteringConfig:
    return ClusteringConfig(
        n_resample=n_resample,
        min_cluster_sizes=(2,),
        min_samples=(1,),
        selection_methods=("eom",),
        score=HDBSCANScoreConfig(min_clusters=99),
        kmeans_k_min=2,
        kmeans_k_max=2,
        random_state=7,
    )


def test_preparation_uses_final_inbound_crossing_and_one_runway_partition() -> None:
    # This track enters once, leaves the terminal area, then makes its final
    # inbound entry.  Release must be reconstructed on the latter segment.
    raw = _raw_track("ACCEPT", north_nm=[60.0, 40.0, 55.0, 45.0, 1.0])
    arrivals = [
        _arrival("ACCEPT", event_time_s=240.0),
        _arrival("MISSING", event_time_s=240.0),
        _arrival("OTHER_RUNWAY", event_time_s=240.0, runway="RW18R"),
    ]

    result = prepare_adsb_tracks_for_clustering(
        arrivals,
        [raw],
        airport=AIRPORT,
        runway="35C",
        config=_pipeline_config(n_resample=13),
    )

    assert result.runway == RUNWAY
    assert result.selected_arrival_count == 2
    assert [item.flight_id for item in result.prepared_tracks] == ["ACCEPT"]
    assert result.rejections[0].flight_id == "MISSING"
    assert result.rejections[0].reason is TrackRejectionReason.MISSING_RAW_TRACK

    prepared = result.prepared_tracks[0]
    assert prepared.raw_track is raw
    assert prepared.terminal_entry.segment_index == 2
    assert prepared.terminal_entry.segment_fraction == pytest.approx(0.5)
    assert prepared.observed_release_time_s == pytest.approx(150.0)
    assert len(prepared.resampled_points_m) == 13
    assert np.linalg.norm(prepared.aligned_points_m[0]) == pytest.approx(50.0 * M_PER_NM)
    np.testing.assert_allclose(prepared.aligned_points_m[-1], [0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(prepared.resampled_points_m[-1], [0.0, 0.0], atol=1.0e-7)
    assert not prepared.resampled_points_m.flags.writeable


def test_preparation_rejects_missing_crossing_and_endpoint_outside_capture_radius() -> None:
    no_crossing = _raw_track("NO_CROSSING", north_nm=[45.0, 30.0, 1.0])
    missed_runway = _raw_track("MISSED_RUNWAY", north_nm=[60.0, 40.0, 3.0])
    arrivals = [
        _arrival("NO_CROSSING", event_time_s=120.0),
        _arrival("MISSED_RUNWAY", event_time_s=120.0),
    ]

    result = prepare_adsb_tracks_for_clustering(
        arrivals,
        [no_crossing, missed_runway],
        airport=AIRPORT,
        runway=RUNWAY,
        config=_pipeline_config(),
    )

    assert not result.prepared_tracks
    reasons = {item.flight_id: item.reason for item in result.rejections}
    assert reasons == {
        "MISSED_RUNWAY": TrackRejectionReason.ENDPOINT_OUTSIDE_CAPTURE_RADIUS,
        "NO_CROSSING": TrackRejectionReason.NO_INBOUND_CROSSING,
    }


def test_catalog_event_clips_later_raw_samples_before_crossing_selection() -> None:
    raw = _raw_track(
        "EVENT_CLIP",
        north_nm=[60.0, 40.0, 55.0, 45.0, 1.0],
    )
    # The event is at the second sample.  The later re-entry must not be used as
    # an observed release for this catalog landing record.
    arrival = _arrival("EVENT_CLIP", event_time_s=60.0)

    result = prepare_adsb_tracks_for_clustering(
        [arrival],
        [raw],
        airport=AIRPORT,
        runway=RUNWAY,
        config=_pipeline_config(),
    )

    assert not result.prepared_tracks
    assert (
        result.rejections[0].reason
        is TrackRejectionReason.ENDPOINT_OUTSIDE_CAPTURE_RADIUS
    )


def test_medoid_speed_is_derived_before_exact_threshold_alignment() -> None:
    raw = _raw_track(
        "SPEED_SOURCE",
        north_nm=[51.0, 49.0, 45.0, 40.0, 20.0, 0.5],
    )
    arrival = _arrival("SPEED_SOURCE", event_time_s=300.0)
    prepared = prepare_adsb_tracks_for_clustering(
        [arrival],
        [raw],
        airport=AIRPORT,
        runway=RUNWAY,
        config=_pipeline_config(),
    ).prepared_tracks[0]

    medoid = prepared.to_medoid_track()

    assert medoid.ground_speed_mps is not None
    assert medoid.lat_deg[-1] == pytest.approx(THRESHOLD_LAT_DEG)
    assert medoid.lon_deg[-1] == pytest.approx(THRESHOLD_LON_DEG)
    # The raw final point is snapped by 0.5 NM at the same timestamp.  Speed
    # provenance must not interpret that alignment correction as motion.
    assert float(medoid.ground_speed_mps[-1]) <= 180.0


def test_adsb_builder_records_release_rejections_and_raw_medoid_sources() -> None:
    arrivals: list[CatalogArrival] = []
    raw_tracks: list[RawADSBTrack] = []
    source_by_id: dict[str, RawADSBTrack] = {}
    for index, lateral_offset in enumerate((-0.10, 0.0, 0.10)):
        flight_id = f"A{index}"
        arrivals.append(_arrival(flight_id, event_time_s=180.0))
        track = _raw_track(
            flight_id,
            north_nm=[60.0, 40.0, 20.0, 0.5],
            east_nm=[lateral_offset] * 4,
        )
        raw_tracks.append(track)
        source_by_id[flight_id] = track
    for index, lateral_offset in enumerate((-0.10, 0.0, 0.10)):
        flight_id = f"B{index}"
        arrivals.append(_arrival(flight_id, event_time_s=180.0))
        track = _raw_track(
            flight_id,
            north_nm=[60.0, 40.0, 20.0, 0.5],
            east_nm=[8.0 + lateral_offset, 7.0 + lateral_offset, 4.0, 0.5],
        )
        raw_tracks.append(track)
        source_by_id[flight_id] = track
    arrivals.append(_arrival("MISSING", event_time_s=180.0))

    result = build_cluster_library_from_adsb(
        arrivals,
        raw_tracks,
        dataset_id="synthetic-adsb",
        airport=AIRPORT,
        runway="35C",
        config=_pipeline_config(n_resample=9),
        metadata={"experiment": "test"},
    )

    library = result.cluster_library
    assert result.library is result.artifact
    assert library.resample_station_count == 9
    assert len(library.assignments) == 6
    assert library.metadata["experiment"] == "test"
    assert set(library.metadata["observed_release_times_s"]) == set(source_by_id)
    assert library.metadata["track_rejections"][0]["flight_id"] == "MISSING"
    assert library.metadata["track_rejections"][0]["reason"] == "missing_raw_track"
    assert library.metadata["adsb_pipeline"]["accepted_track_count"] == 6
    assert library.metadata["adsb_pipeline"]["rejected_track_count"] == 1
    assert library.metadata["station_order"] == "upstream_to_threshold"

    medoid_ids = {item.medoid_flight_id for item in library.medoids}
    assert set(result.raw_medoid_tracks) == medoid_ids
    assert all(result.raw_medoid_tracks[item] is source_by_id[item] for item in medoid_ids)
    assert dict(result.observed_release_times_s) == library.metadata["observed_release_times_s"]
    assert all(0.0 < value < 60.0 for value in result.observed_release_times_s.values())
    for medoid in result.template_medoid_tracks.values():
        assert medoid.time_s[0] in result.observed_release_times_s.values()
        assert medoid.lat_deg[-1] == pytest.approx(THRESHOLD_LAT_DEG)
        assert medoid.lon_deg[-1] == pytest.approx(THRESHOLD_LON_DEG)
        if medoid.ground_speed_mps is not None:
            assert np.all(medoid.ground_speed_mps > 0.0)
