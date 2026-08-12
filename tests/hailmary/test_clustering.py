from __future__ import annotations

import json

import numpy as np
import pytest

from hailmary.config import ClusteringConfig, HDBSCANScoreConfig
from hailmary.clustering import (
    ClusterLibrary,
    HDBSCANSelectionConfig,
    MedoidRecord,
    assign_all_flights,
    assign_new_flights,
    build_cluster_library,
    compute_cluster_diagnostics,
    fit_shape_features,
    run_hdbscan_sweep,
    select_medoid,
    transform_shape_features,
)
from hailmary.geometry import (
    LocalFrame,
    orient_upstream_to_threshold,
    prepare_track_for_clustering,
    resample_polyline,
    to_executable_station_order,
)


def _two_cloud_tracks() -> dict[str, np.ndarray]:
    stations = np.linspace(0.0, 30_000.0, 8)
    tracks: dict[str, np.ndarray] = {}
    for index, offset in enumerate((-120.0, 0.0, 100.0)):
        tracks[f"A{index}"] = np.column_stack((np.full_like(stations, offset), stations))
    for index, offset in enumerate((-100.0, 0.0, 120.0)):
        tracks[f"B{index}"] = np.column_stack((8_000.0 + np.full_like(stations, offset), stations))
    return tracks


def _forced_fallback_config() -> HDBSCANSelectionConfig:
    return HDBSCANSelectionConfig(
        min_cluster_sizes=(2,),
        min_samples_values=(1,),
        cluster_selection_methods=("eom",),
        minimum_cluster_count=99,
        kmeans_k_values=(2,),
        random_state=7,
    )


def test_local_frame_round_trip_and_runway_origin() -> None:
    frame = LocalFrame(32.9, -97.0)
    lat = np.asarray([32.9, 32.91])
    lon = np.asarray([-97.0, -96.99])

    east, north = frame.project(lat, lon)
    round_lat, round_lon = frame.unproject(east, north)

    assert float(east[0]) == pytest.approx(0.0, abs=1.0e-8)
    assert float(north[0]) == pytest.approx(0.0, abs=1.0e-8)
    np.testing.assert_allclose(round_lat, lat, atol=1.0e-9)
    np.testing.assert_allclose(round_lon, lon, atol=1.0e-9)


def test_resampling_normalizes_direction_then_builds_increasing_remaining_distance() -> None:
    threshold_to_upstream = np.asarray([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0]])

    oriented = orient_upstream_to_threshold(threshold_to_upstream)
    sampled = resample_polyline(oriented, n_points=8)
    executable = to_executable_station_order(sampled.points_m)

    np.testing.assert_allclose(sampled.points_m[0], [4.0, 3.0])
    np.testing.assert_allclose(sampled.points_m[-1], [0.0, 0.0])
    np.testing.assert_allclose(sampled.distance_m, np.linspace(0.0, 7.0, 8))
    assert executable.s_m[0] == 0.0
    assert np.all(np.diff(executable.s_m) > 0.0)
    assert not executable.points_m.flags.writeable


def test_terminal_preparation_interpolates_radius_and_appends_exact_threshold() -> None:
    track = np.asarray([[0.0, 60.0], [0.0, 40.0], [0.0, 1.0]]) * 1_852.0

    sampled = prepare_track_for_clustering(
        track,
        terminal_radius_m=50.0 * 1_852.0,
        threshold_capture_radius_m=2.0 * 1_852.0,
        n_points=5,
    )

    assert np.linalg.norm(sampled.points_m[0]) == pytest.approx(50.0 * 1_852.0)
    np.testing.assert_allclose(sampled.points_m[-1], [0.0, 0.0])
    assert len(sampled.points_m) == 5


def test_training_transform_is_reused_for_held_out_tracks_and_arrays_are_read_only() -> None:
    training = _two_cloud_tracks()
    fitted = fit_shape_features(training)
    held_out = {"NEW": training["A0"] + np.asarray([20.0, 0.0])}

    transformed = transform_shape_features(held_out, fitted.transform)
    expected = (transformed.raw - fitted.mean) / fitted.scale

    assert fitted.track_ids == tuple(sorted(training))
    np.testing.assert_allclose(transformed.standardized, expected)
    assert not fitted.raw.flags.writeable
    assert not fitted.transform.mean.flags.writeable
    with pytest.raises(ValueError):
        fitted.raw[0, 0] = 123.0


def test_medoid_tie_breaks_by_stable_flight_id_and_is_an_observed_member() -> None:
    tracks = np.asarray(
        [
            [[0.0, 1.0], [1.0, 1.0]],
            [[0.0, -1.0], [1.0, -1.0]],
        ]
    )

    record = select_medoid(["B", "A"], tracks)

    assert record.medoid_flight_id == "A"
    assert record.medoid_flight_id in record.member_flight_ids
    assert not record.points_m.flags.writeable


def test_noise_is_assigned_to_nearest_medoid_with_ood_provenance_and_stable_tie() -> None:
    zero = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    ten = np.asarray([[10.0, 0.0], [11.0, 0.0]])
    medoids = (
        MedoidRecord(0, "M0", ("M0",), zero, 0.0, 0.0, 1.0, 0.0),
        MedoidRecord(1, "M1", ("M1",), ten, 0.0, 0.0, 1.0, 0.0),
    )
    midway = np.asarray([[5.0, 0.0], [6.0, 0.0]])

    assignments = assign_all_flights(["NOISE"], np.asarray([midway]), [-1], [0.0], medoids)

    assert assignments[0].cluster_id == 0
    assert assignments[0].was_hdbscan_noise
    assert assignments[0].out_of_distribution
    assert not assignments[0].included_in_template_training


def test_cluster_diagnostics_report_physical_dispersion_and_robust_bearing_span() -> None:
    bearings_deg = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 180.0])
    radius_m = 10_000.0
    starts = np.column_stack(
        (
            radius_m * np.sin(np.deg2rad(bearings_deg)),
            radius_m * np.cos(np.deg2rad(bearings_deg)),
        )
    )
    tracks = np.stack(
        [np.vstack((start, np.zeros(2))) for start in starts],
        axis=0,
    )

    diagnostics = compute_cluster_diagnostics(
        np.zeros(len(tracks), dtype=np.int64),
        np.linspace(0.5, 1.0, len(tracks)),
        tracks,
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.member_count == 6
    # Ninety percent of six members retains all six, so the remote bearing is
    # intentionally visible. Larger operational clusters may ignore their
    # weakest ten-percent tail.
    assert diagnostic.entry_bearing_p90_span_deg == pytest.approx(170.0)
    assert diagnostic.centroid_distance_p90_m > 0.0
    assert diagnostic.membership_probability_median == pytest.approx(0.75)


def test_rejected_hdbscan_sweep_uses_deterministic_kmeans_fallback() -> None:
    features = fit_shape_features(_two_cloud_tracks())
    config = _forced_fallback_config()

    first = run_hdbscan_sweep(features.standardized, config)
    second = run_hdbscan_sweep(features.standardized, config)

    assert first.algorithm == "kmeans"
    assert first.used_fallback
    assert first.parameter_dict["k"] == 2
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.to_dict() == second.to_dict()
    assert set(first.labels) == {0, 1}


def test_runner_accepts_package_level_clustering_config() -> None:
    features = fit_shape_features(_two_cloud_tracks())
    config = ClusteringConfig(
        n_resample=8,
        min_cluster_sizes=(2,),
        min_samples=(1,),
        selection_methods=("eom",),
        score=HDBSCANScoreConfig(min_clusters=99),
        kmeans_k_min=2,
        kmeans_k_max=2,
    )

    result = run_hdbscan_sweep(features.standardized, config)

    assert result.algorithm == "kmeans"
    assert result.parameter_dict["k"] == 2


def test_cluster_library_is_canonical_round_trippable_and_assigns_every_flight() -> None:
    tracks = _two_cloud_tracks()
    frame = LocalFrame(32.9, -97.0)

    first = build_cluster_library(
        tracks,
        dataset_id="synthetic",
        airport="KDFW",
        runway="35C",
        projection=frame.to_dict(),
        config=_forced_fallback_config(),
    )
    second = build_cluster_library(
        dict(reversed(list(tracks.items()))),
        dataset_id="synthetic",
        airport="KDFW",
        runway="RW35C",
        projection=frame.to_dict(),
        config=_forced_fallback_config(),
    )
    restored = ClusterLibrary.from_json(first.to_json())

    assert first.artifact_content_hash == second.artifact_content_hash
    assert restored.to_json() == first.to_json()
    assert len(first.assignments) == len(tracks)
    assert all(item.cluster_id >= 0 for item in first.assignments)
    assert first.corpus_assignments == first.assignments
    assert not first.hdbscan_outlier_assignments
    assert all(item.medoid_flight_id in item.member_flight_ids for item in first.medoids)
    assert json.loads(first.to_json())["artifact_content_hash"] == first.artifact_content_hash


def test_cluster_library_excludes_hdbscan_noise_from_corpus_assignments() -> None:
    station = np.linspace(10_000.0, 0.0, 16)
    tracks: dict[str, np.ndarray] = {}
    for prefix, base in (("A", -5_000.0), ("B", 5_000.0)):
        for index, offset in enumerate((-50.0, 0.0, 50.0, 100.0)):
            tracks[f"{prefix}{index}"] = np.column_stack(
                (np.full_like(station, base + offset), station)
            )
    tracks["NOISE"] = np.column_stack((np.linspace(30_000.0, 0.0, 16), station))
    config = HDBSCANSelectionConfig(
        min_cluster_sizes=(3,),
        min_samples_values=(2,),
        cluster_selection_methods=("eom",),
        max_noise_fraction=1.0,
        max_fragmentation=1.0,
        max_entry_bearing_span_deg=360.0,
    )

    library = build_cluster_library(
        tracks,
        dataset_id="synthetic-noise",
        airport="KDFW",
        runway="35C",
        projection=LocalFrame(32.9, -97.0).to_dict(),
        config=config,
    )

    assert [item.flight_id for item in library.hdbscan_outlier_assignments] == ["NOISE"]
    assert "NOISE" not in {item.flight_id for item in library.corpus_assignments}
    assert len(library.corpus_assignments) == len(tracks) - 1


def test_cluster_library_detects_hash_tampering() -> None:
    artifact = build_cluster_library(
        _two_cloud_tracks(),
        dataset_id="synthetic",
        airport="KDFW",
        runway="35C",
        projection=LocalFrame(32.9, -97.0).to_dict(),
        config=_forced_fallback_config(),
    )
    payload = artifact.to_dict()
    payload["dataset_id"] = "tampered"

    with pytest.raises(ValueError, match="content hash"):
        ClusterLibrary.from_dict(payload)


def test_portable_kmeans_predictor_assigns_new_flight_with_membership() -> None:
    tracks = _two_cloud_tracks()
    library = build_cluster_library(
        tracks,
        dataset_id="synthetic",
        airport="KDFW",
        runway="35C",
        projection=LocalFrame(32.9, -97.0).to_dict(),
        config=_forced_fallback_config(),
    )
    restored = ClusterLibrary.from_json(library.to_json())
    expected_cluster = next(
        item.cluster_id for item in restored.assignments if item.flight_id == "A0"
    )

    assignment = assign_new_flights(
        ["HELD"],
        np.asarray([tracks["A0"] + np.asarray([5.0, 0.0])]),
        restored,
    )[0]

    assert assignment.cluster_id == expected_cluster
    assert assignment.membership_probability == 1.0
    assert not assignment.was_hdbscan_noise
    assert restored.prediction_training is not None
    assert not restored.prediction_training.standardized_features.flags.writeable


def test_portable_hdbscan_predictor_uses_approximate_membership() -> None:
    pytest.importorskip("hdbscan")
    tracks = _two_cloud_tracks()
    config = HDBSCANSelectionConfig(
        min_cluster_sizes=(2,),
        min_samples_values=(1,),
        cluster_selection_methods=("eom",),
        max_noise_fraction=1.0,
        max_fragmentation=1.0,
        minimum_cluster_count=2,
        kmeans_k_values=(2,),
        random_state=7,
    )
    library = build_cluster_library(
        tracks,
        dataset_id="synthetic",
        airport="KDFW",
        runway="35C",
        projection=LocalFrame(32.9, -97.0).to_dict(),
        config=config,
    )
    assert library.clustering.algorithm == "hdbscan"
    expected_cluster = next(
        item.cluster_id for item in library.assignments if item.flight_id == "A1"
    )

    assignment = assign_new_flights(
        ["HELD"],
        np.asarray([tracks["A1"] + np.asarray([1.0, 0.0])]),
        library,
    )[0]

    assert assignment.cluster_id == expected_cluster
    assert assignment.membership_probability > 0.0
    assert not assignment.was_hdbscan_noise
