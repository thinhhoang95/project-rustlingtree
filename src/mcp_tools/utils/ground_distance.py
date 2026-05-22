from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from openap import aero

from simap.backends import EffectivePolarBackend
from simap.calibration import build_default_aircraft_config, suggest_approach_mass_kg
from simap.config import AircraftConfig
from simap.fms import FMSRequest, FMSSpeedTargets, plan_fms_descent
from simap.openap_adapter import OpenAPObjects, extract_aircraft_data, load_openap, openap_dT
from simap.path_geometry import ReferencePath
from simap.units import ft_to_m, kts_to_mps, m_to_ft, mps_to_kts
from simap.weather import ConstantWeather

_NMI_TO_M = 1_852.0
_SYNTHETIC_THRESHOLD_LAT_DEG = 32.8968
_SYNTHETIC_THRESHOLD_LON_DEG = -97.0380


@dataclass(frozen=True)
class GroundDistanceEstimate:
    required_ground_distance_m: float
    required_ground_distance_nmi: float
    success: bool
    message: str
    initial_cas_kts: float
    descent_time_s: float | None
    total_time_s: float
    tod_distance_m: float | None
    tod_distance_nmi: float | None
    final_altitude_ft: float
    final_cas_kts: float


@dataclass(frozen=True)
class _AircraftRuntime:
    cfg: AircraftConfig
    openap: OpenAPObjects
    perf: EffectivePolarBackend
    reference_path: ReferencePath


@dataclass
class GroundDistanceEstimationTool:
    aircraft_type: str = "A320"
    payload_kg: float = 12_000.0
    runway_altitude_ft: float = 620.0
    dt_s: float = 1.0
    max_time_s: float = 7_200.0
    synthetic_path_nmi: float = 600.0
    tod_tolerance_m: float = 5.0
    max_tod_iterations: int = 40
    _runtime: _AircraftRuntime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.payload_kg < 0.0:
            raise ValueError("payload_kg must be nonnegative")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.max_time_s <= 0.0:
            raise ValueError("max_time_s must be positive")
        if self.synthetic_path_nmi <= 0.0:
            raise ValueError("synthetic_path_nmi must be positive")
        if self.tod_tolerance_m <= 0.0:
            raise ValueError("tod_tolerance_m must be positive")
        if self.max_tod_iterations <= 0:
            raise ValueError("max_tod_iterations must be positive")

    def __call__(self, altitude_ft: float, tas_kts: float) -> float:
        return self.estimate(altitude_ft=altitude_ft, tas_kts=tas_kts).required_ground_distance_nmi

    def estimate(self, altitude_ft: float, tas_kts: float) -> GroundDistanceEstimate:
        altitude_ft = self._finite_float(altitude_ft, name="altitude_ft")
        tas_kts = self._finite_float(tas_kts, name="tas_kts")
        if tas_kts <= 0.0:
            raise ValueError("tas_kts must be positive")
        if altitude_ft <= self.runway_altitude_ft:
            raise ValueError("altitude_ft must be greater than runway_altitude_ft")

        runtime = self._aircraft_runtime()
        start_h_m = ft_to_m(altitude_ft)
        target_h_m = ft_to_m(self.runway_altitude_ft)
        start_tas_mps = kts_to_mps(tas_kts)
        start_cas_mps = float(aero.tas2cas(start_tas_mps, start_h_m, dT=openap_dT(0.0)))

        speed_targets = FMSSpeedTargets(
            clean_cas_mps=start_cas_mps,
            approach_cas_mps=float(runtime.openap.wrap.finalapp_vcas()["default"]),
            final_cas_mps=float(runtime.openap.wrap.landing_speed()["default"]),
        )
        request = FMSRequest(
            cfg=runtime.cfg,
            perf=runtime.perf,
            reference_path=runtime.reference_path,
            start_s_m=runtime.reference_path.total_length_m,
            start_h_m=start_h_m,
            start_cas_mps=start_cas_mps,
            target_h_m=target_h_m,
            speed_targets=speed_targets,
            weather=ConstantWeather(),
            dt_s=self.dt_s,
            max_time_s=self.max_time_s,
        )
        result = plan_fms_descent(
            request,
            tod_tolerance_m=self.tod_tolerance_m,
            max_tod_iterations=self.max_tod_iterations,
        )

        required_ground_distance_m = float(
            result.descent_segment_distance_m
            if result.descent_segment_distance_m is not None
            else result.descent_distance_m
        )
        tod_distance_nmi = None if result.tod_s_m is None else float(result.tod_s_m / _NMI_TO_M)
        return GroundDistanceEstimate(
            required_ground_distance_m=required_ground_distance_m,
            required_ground_distance_nmi=float(required_ground_distance_m / _NMI_TO_M),
            success=bool(result.success),
            message=result.message,
            initial_cas_kts=float(mps_to_kts(start_cas_mps)),
            descent_time_s=result.descent_segment_time_s,
            total_time_s=float(result.descent_time_s),
            tod_distance_m=result.tod_s_m,
            tod_distance_nmi=tod_distance_nmi,
            final_altitude_ft=float(m_to_ft(result.h_m[-1])),
            final_cas_kts=float(mps_to_kts(result.v_cas_mps[-1])),
        )

    def _aircraft_runtime(self) -> _AircraftRuntime:
        if self._runtime is None:
            openap_objects = load_openap(self.aircraft_type)
            aircraft_data = extract_aircraft_data(openap_objects)
            mass_kg = suggest_approach_mass_kg(aircraft_data, payload_kg=self.payload_kg)
            cfg, openap_objects = build_default_aircraft_config(
                self.aircraft_type,
                mass_kg=mass_kg,
                openap_objects=openap_objects,
            )
            perf = EffectivePolarBackend(cfg=cfg, openap=openap_objects)
            self._runtime = _AircraftRuntime(
                cfg=cfg,
                openap=openap_objects,
                perf=perf,
                reference_path=self._build_synthetic_path(),
            )
        return self._runtime

    def _build_synthetic_path(self) -> ReferencePath:
        latitude_delta_deg = (self.synthetic_path_nmi * _NMI_TO_M) / 1_000.0 / 111.0
        return ReferencePath.from_geographic(
            lat_deg=np.asarray(
                [
                    _SYNTHETIC_THRESHOLD_LAT_DEG + latitude_delta_deg,
                    _SYNTHETIC_THRESHOLD_LAT_DEG,
                ],
                dtype=float,
            ),
            lon_deg=np.asarray(
                [
                    _SYNTHETIC_THRESHOLD_LON_DEG,
                    _SYNTHETIC_THRESHOLD_LON_DEG,
                ],
                dtype=float,
            ),
        )

    @staticmethod
    def _finite_float(value: float, *, name: str) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
