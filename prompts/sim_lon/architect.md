I would build it as a **small package with one strict rule: all internal state is SI**. That matters because OpenAP is mixed: `openap.aero` uses SI units, WRAP returns kinematic parameters in km / m/s / m²/s, while the drag and thrust classes take TAS in knots, altitude in feet, vertical rate in ft/min, and mass in kilograms. Centralizing those conversions in one adapter file will save you a lot of debugging. OpenAP can be installed from PyPI with `pip install --upgrade openap`, and its documented modules for your use case are `prop`, `aero`, `WRAP`, `Drag`, `Thrust`, and optionally `FlightGenerator`. ([OpenAP][1])

**Note that the following serves more as guidelines. You should verify the plan carefully before implementation.**

```text
approachsim/
  __init__.py
  config.py          # dataclasses for aircraft, modes, and sim settings
  units.py           # ONLY place that converts SI <-> OpenAP performance units
  openap_adapter.py  # exact OpenAP calls; loads aircraft/engine/WRAP/drag/thrust
  profiles.py        # speed schedule, altitude path, centerline interpolation
  weather.py         # along-track wind model
  backends.py        # PerformanceBackend protocol + OpenAP backend
  dynamics.py        # state derivative function
  simulator.py       # fixed-step integration loop and logging
  examples/
    run_a320.py
```

The architecture split I recommend is:

* `openap_adapter.py`: exact API calls to OpenAP and no model logic.
* `profiles.py`: path and schedule interpolation only.
* `backends.py`: force/limit models behind a clean interface.
* `dynamics.py`: the ODE right-hand side.
* `simulator.py`: time stepping and output assembly.

That layout keeps your research logic separate from OpenAP plumbing, which makes it easy to swap to a sovereign coefficient backend later. ([OpenAP][1])

## 1) `config.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModeName = Literal["clean", "approach", "final"]


@dataclass(frozen=True)
class ModeConfig:
    name: ModeName
    tau_v_s: float
    vs_min_mps: float
    vs_max_mps: float
    # Effective drag model for your reduced-order simulator.
    # Start with clean from OpenAP and tune approach/final later.
    cd0: float
    k: float


@dataclass(frozen=True)
class AircraftConfig:
    typecode: str
    engine_name: str
    mass_kg: float
    wing_area_m2: float
    vmo_kts: float
    mmo: float
    clean: ModeConfig
    approach: ModeConfig
    final: ModeConfig
    k_h_sinv: float = 0.03     # altitude-path capture gain
    a_acc_max_mps2: float = 0.8
    final_gate_m: float = 12_000.0
    approach_gate_m: float = 35_000.0
```

## 2) `units.py`

```python
from __future__ import annotations

from openap import aero


def m_to_ft(x_m: float) -> float:
    return x_m / aero.ft


def ft_to_m(x_ft: float) -> float:
    return x_ft * aero.ft


def mps_to_kts(x_mps: float) -> float:
    return x_mps / aero.kts


def kts_to_mps(x_kts: float) -> float:
    return x_kts * aero.kts


def mps_to_fpm(x_mps: float) -> float:
    return x_mps / aero.fpm


def fpm_to_mps(x_fpm: float) -> float:
    return x_fpm * aero.fpm


def km_to_m(x_km: float) -> float:
    return x_km * 1000.0
```

## 3) `openap_adapter.py`

OpenAP’s documented calls for your model are:

* `prop.aircraft("A320")` for aircraft data like `drag`, `wing`, `oew`, `mlw`, `mtow`, `vmo`, and `mmo`,
* `prop.engine(...)` for engine data including `max_thrust`,
* `WRAP(ac="A320")` for phase parameters like `descent_const_vcas`, `finalapp_vcas`, `finalapp_vs`, `landing_speed`,
* `Drag(ac="A320")` and `Thrust(ac="A320", eng=...)` for force calculations. ([OpenAP][2])

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from openap import prop
from openap.drag import Drag
from openap.thrust import Thrust
from openap.kinematic import WRAP

from .config import AircraftConfig, ModeConfig


@dataclass(frozen=True)
class OpenAPObjects:
    aircraft: dict[str, Any]
    engine: dict[str, Any]
    wrap: WRAP
    drag: Drag
    thrust: Thrust


def load_openap(typecode: str) -> OpenAPObjects:
    ac = prop.aircraft(typecode)
    engine_name = ac["engine"]["default"]
    eng = prop.engine(engine_name)
    return OpenAPObjects(
        aircraft=ac,
        engine=eng,
        wrap=WRAP(ac=typecode),
        drag=Drag(ac=typecode),
        thrust=Thrust(ac=typecode, eng=engine_name),
    )


def wrap_default(wrap: WRAP, method_name: str) -> float:
    params = getattr(wrap, method_name)()
    return float(params["default"])


def wrap_sample(wrap: WRAP, method_name: str, rng: np.random.Generator) -> float:
    params = getattr(wrap, method_name)()
    model_class = getattr(stats, params["statmodel"])
    model = model_class(*params["statmodel_params"])
    x = model.rvs(random_state=rng)
    return float(np.clip(x, params["minimum"], params["maximum"]))


def build_aircraft_config(typecode: str, payload_kg: float = 12_000.0) -> tuple[AircraftConfig, OpenAPObjects]:
    o = load_openap(typecode)
    ac = o.aircraft

    # Reasonable approach mass policy:
    # clip OEW + payload to not exceed MLW
    mass_kg = min(ac["mlw"], ac["oew"] + payload_kg)

    cd0_clean = float(ac["drag"]["cd0"])
    k_clean = float(ac["drag"]["k"])

    # Start simple: use clean coefficients as baseline.
    # Tune approach/final effective coefficients later against data.
    clean = ModeConfig(
        name="clean",
        tau_v_s=18.0,
        vs_min_mps=-12.0,
        vs_max_mps=3.0,
        cd0=cd0_clean,
        k=k_clean,
    )
    approach = ModeConfig(
        name="approach",
        tau_v_s=22.0,
        vs_min_mps=-10.0,
        vs_max_mps=2.0,
        cd0=cd0_clean * 1.35,
        k=k_clean * 1.10,
    )
    final = ModeConfig(
        name="final",
        tau_v_s=25.0,
        vs_min_mps=-8.0,
        vs_max_mps=1.5,
        cd0=cd0_clean * 1.75,
        k=k_clean * 1.20,
    )

    cfg = AircraftConfig(
        typecode=typecode,
        engine_name=ac["engine"]["default"],
        mass_kg=mass_kg,
        wing_area_m2=float(ac["wing"]["area"]),
        vmo_kts=float(ac["vmo"]),
        mmo=float(ac["mmo"]),
        clean=clean,
        approach=approach,
        final=final,
    )
    return cfg, o
```

The `wrap_default()` and `wrap_sample()` helpers follow the documented WRAP return format: each WRAP method returns a dict with `default`, `minimum`, `maximum`, `statmodel`, and `statmodel_params`, and the handbook’s own example uses `scipy.stats` via `getattr(stats, params["statmodel"])`. WRAP’s kinematic values are documented in km / m/s / m²/s. ([OpenAP][3])

## 4) `profiles.py`

I strongly recommend expressing the approach in **distance remaining to threshold**, because that makes altitude and speed schedules much cleaner in code. So I switch from your earlier `s` to `x_m`, where `x_m = 0` at threshold and increases outbound.

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScalarProfile:
    x_m: np.ndarray  # ascending: 0 at threshold, larger outbound
    y: np.ndarray

    def value(self, x_m: float) -> float:
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        return float(np.interp(x, self.x_m, self.y))

    def slope(self, x_m: float) -> float:
        # Finite-difference slope dy/dx
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        i = int(np.searchsorted(self.x_m, x))
        i = max(1, min(i, len(self.x_m) - 1))
        dx = self.x_m[i] - self.x_m[i - 1]
        dy = self.y[i] - self.y[i - 1]
        return float(dy / dx)


@dataclass(frozen=True)
class Centerline:
    x_m: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray

    def latlon(self, x_m: float) -> tuple[float, float]:
        x = float(np.clip(x_m, self.x_m[0], self.x_m[-1]))
        lat = float(np.interp(x, self.x_m, self.lat_deg))
        lon = float(np.interp(x, self.x_m, self.lon_deg))
        return lat, lon


def build_simple_glidepath(
    threshold_elevation_m: float,
    intercept_distance_m: float,
    intercept_altitude_m: float,
    glide_deg: float = 3.0,
    n: int = 400,
) -> ScalarProfile:
    x = np.linspace(0.0, intercept_distance_m, n)
    # Straight 3-degree line ending at threshold elevation.
    h = threshold_elevation_m + np.tan(np.deg2rad(glide_deg)) * x
    # Optionally cap above the intercept altitude.
    h = np.minimum(h, intercept_altitude_m)
    return ScalarProfile(x_m=x, y=h)
```

And this is the first speed schedule I would build from WRAP:

```python
from __future__ import annotations

import numpy as np

from .profiles import ScalarProfile
from .units import km_to_m


def build_default_speed_schedule_from_wrap(wrap) -> ScalarProfile:
    v_des_mps = float(wrap.descent_const_vcas()["default"])
    v_final_mps = float(wrap.finalapp_vcas()["default"])
    v_land_mps = float(wrap.landing_speed()["default"])

    # These distance nodes are YOUR policy, not OpenAP's.
    # Start here, then tune with data.
    x_km = np.array([0.0, 8.0, 30.0, 60.0], dtype=float)
    v_mps = np.array([v_land_mps, v_final_mps, v_des_mps, v_des_mps], dtype=float)
    return ScalarProfile(x_m=km_to_m(x_km), y=v_mps)
```

Using `descent_const_vcas`, `finalapp_vcas`, and `landing_speed` here matches the WRAP API documented for descent, final approach, and landing. ([OpenAP][3])

## 5) `weather.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantWind:
    alongtrack_mps: float = 0.0
    delta_isa_K: float = 0.0

    def alongtrack(self, x_m: float, h_m: float, t_s: float) -> float:
        return self.alongtrack_mps
```

## 6) `backends.py`

The best pattern is a protocol plus one concrete OpenAP-backed implementation.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openap import aero

from .config import AircraftConfig, ModeName
from .openap_adapter import OpenAPObjects
from .units import m_to_ft, mps_to_fpm, mps_to_kts


class PerformanceBackend(Protocol):
    def drag_newtons(
        self,
        mode: ModeName,
        mass_kg: float,
        v_tas_mps: float,
        h_m: float,
        vs_mps: float,
    ) -> float: ...

    def idle_thrust_newtons(self, v_tas_mps: float, h_m: float) -> float: ...


@dataclass
class OpenAPReferenceBackend:
    cfg: AircraftConfig
    openap: OpenAPObjects

    def drag_newtons(
        self,
        mode: ModeName,
        mass_kg: float,
        v_tas_mps: float,
        h_m: float,
        vs_mps: float,
    ) -> float:
        tas_kts = mps_to_kts(v_tas_mps)
        alt_ft = m_to_ft(h_m)
        vs_fpm = mps_to_fpm(vs_mps)

        if mode == "clean":
            return float(
                self.openap.drag.clean(
                    mass=mass_kg,
                    tas=tas_kts,
                    alt=alt_ft,
                    vs=vs_fpm,
                )
            )

        # Internal mapping only, for a reference backend.
        # Replace later with an effective-polar backend if desired.
        if mode == "approach":
            return float(
                self.openap.drag.nonclean(
                    mass=mass_kg,
                    tas=tas_kts,
                    alt=alt_ft,
                    flap_angle=15.0,
                    vs=vs_fpm,
                    landing_gear=False,
                )
            )

        return float(
            self.openap.drag.nonclean(
                mass=mass_kg,
                tas=tas_kts,
                alt=alt_ft,
                flap_angle=30.0,
                vs=vs_fpm,
                landing_gear=True,
            )
        )

    def idle_thrust_newtons(self, v_tas_mps: float, h_m: float) -> float:
        return float(
            self.openap.thrust.descent_idle(
                tas=mps_to_kts(v_tas_mps),
                alt=m_to_ft(h_m),
            )
        )
```

This backend uses the documented `Drag.clean()`, `Drag.nonclean()`, and `Thrust.descent_idle()` signatures directly. The handbook examples show `drag.clean(mass=..., tas=..., alt=..., vs=...)`, `drag.nonclean(..., flap_angle=..., landing_gear=True)`, and `thrust.descent_idle(tas=..., alt=...)`. ([OpenAP][4])

## 7) `dynamics.py`

This is your actual reduced-order model.

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from openap import aero

from .backends import PerformanceBackend
from .config import AircraftConfig, ModeConfig
from .profiles import ScalarProfile
from .weather import ConstantWind


@dataclass(frozen=True)
class State:
    t_s: float
    x_m: float          # distance remaining to threshold
    h_m: float          # altitude
    v_tas_mps: float    # true airspeed


def mode_for_x(cfg: AircraftConfig, x_m: float) -> ModeConfig:
    if x_m <= cfg.final_gate_m:
        return cfg.final
    if x_m <= cfg.approach_gate_m:
        return cfg.approach
    return cfg.clean


def gamma_from_altitude_profile(h_profile: ScalarProfile, x_m: float) -> float:
    # h = h(x), x positive outbound from threshold.
    # On approach x decreases, but gamma only depends on local slope.
    dh_dx = h_profile.slope(x_m)
    return float(np.arctan(dh_dx))


def rhs(
    state: State,
    cfg: AircraftConfig,
    perf: PerformanceBackend,
    altitude_profile: ScalarProfile,
    speed_schedule_cas: ScalarProfile,
    weather: ConstantWind,
) -> np.ndarray:
    x_m = state.x_m
    h_m = state.h_m
    v_tas = max(1.0, state.v_tas_mps)

    mode = mode_for_x(cfg, x_m)
    wind_mps = weather.alongtrack(x_m, h_m, state.t_s)
    gs_mps = max(1.0, v_tas + wind_mps)

    h_ref_m = altitude_profile.value(x_m)
    gamma_ref_rad = gamma_from_altitude_profile(altitude_profile, x_m)

    hdot_ff = -gs_mps * np.tan(gamma_ref_rad)
    hdot_cmd = hdot_ff + cfg.k_h_sinv * (h_ref_m - h_m)
    hdot_cmd = float(np.clip(hdot_cmd, mode.vs_min_mps, mode.vs_max_mps))

    d_newtons = perf.drag_newtons(
        mode=mode.name,
        mass_kg=cfg.mass_kg,
        v_tas_mps=v_tas,
        h_m=h_m,
        vs_mps=hdot_cmd,
    )
    t_idle_newtons = perf.idle_thrust_newtons(v_tas, h_m)

    v_ref_cas_mps = speed_schedule_cas.value(x_m)
    v_ref_tas_mps = float(aero.cas2tas(v_ref_cas_mps, h_m, dT=weather.delta_isa_K))

    vdot_cmd = (v_ref_tas_mps - v_tas) / mode.tau_v_s

    # Feasible deceleration cap from drag - idle thrust - downhill gravity release
    a_dec_max = max(
        0.0,
        (d_newtons - t_idle_newtons) / cfg.mass_kg - aero.g0 * abs(np.sin(gamma_ref_rad)),
    )

    vdot = float(np.clip(vdot_cmd, -a_dec_max, cfg.a_acc_max_mps2))

    xdot = -gs_mps
    return np.array([xdot, hdot_cmd, vdot], dtype=float)
```

The OpenAP drag formulation is the standard point-mass polar (C_d = C_{d0} + k C_l^2), (D = \tfrac12 \rho v^2 S C_d), with climb/descent effects handled through vertical motion in the lift estimate, and the handbook states that idle thrust is modeled at about 7% of maximum thrust at the same altitude and speed. Those facts are what justify the `a_dec_max` cap. ([OpenAP][4])

## 8) `simulator.py`

```python
from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd
from openap import aero

from .config import AircraftConfig
from .backends import PerformanceBackend
from .dynamics import State, rhs
from .profiles import ScalarProfile, Centerline
from .weather import ConstantWind


class ApproachSimulator:
    def __init__(
        self,
        cfg: AircraftConfig,
        perf: PerformanceBackend,
        altitude_profile: ScalarProfile,
        speed_schedule_cas: ScalarProfile,
        centerline: Centerline | None = None,
        weather: ConstantWind | None = None,
    ) -> None:
        self.cfg = cfg
        self.perf = perf
        self.altitude_profile = altitude_profile
        self.speed_schedule_cas = speed_schedule_cas
        self.centerline = centerline
        self.weather = weather or ConstantWind()

    def step(self, state: State, dt_s: float) -> State:
        # RK4 on [x, h, v]
        y0 = np.array([state.x_m, state.h_m, state.v_tas_mps], dtype=float)

        def f(y: np.ndarray, t: float) -> np.ndarray:
            s = State(t_s=t, x_m=float(y[0]), h_m=float(y[1]), v_tas_mps=float(y[2]))
            return rhs(
                state=s,
                cfg=self.cfg,
                perf=self.perf,
                altitude_profile=self.altitude_profile,
                speed_schedule_cas=self.speed_schedule_cas,
                weather=self.weather,
            )

        k1 = f(y0, state.t_s)
        k2 = f(y0 + 0.5 * dt_s * k1, state.t_s + 0.5 * dt_s)
        k3 = f(y0 + 0.5 * dt_s * k2, state.t_s + 0.5 * dt_s)
        k4 = f(y0 + dt_s * k3, state.t_s + dt_s)
        y1 = y0 + (dt_s / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return State(
            t_s=state.t_s + dt_s,
            x_m=max(0.0, float(y1[0])),
            h_m=max(0.0, float(y1[1])),
            v_tas_mps=max(1.0, float(y1[2])),
        )

    def run(self, initial: State, dt_s: float = 1.0, t_max_s: float = 4_000.0) -> pd.DataFrame:
        rows: list[dict] = []
        s = initial

        while s.t_s <= t_max_s and s.x_m > 1.0:
            wind = self.weather.alongtrack(s.x_m, s.h_m, s.t_s)
            gs = max(1.0, s.v_tas_mps + wind)
            v_cas = float(aero.tas2cas(s.v_tas_mps, s.h_m, dT=self.weather.delta_isa_K))
            h_ref = self.altitude_profile.value(s.x_m)
            v_ref_cas = self.speed_schedule_cas.value(s.x_m)

            row = {
                "t_s": s.t_s,
                "x_m": s.x_m,
                "h_m": s.h_m,
                "v_tas_mps": s.v_tas_mps,
                "v_cas_mps": v_cas,
                "gs_mps": gs,
                "h_ref_m": h_ref,
                "v_ref_cas_mps": v_ref_cas,
            }

            if self.centerline is not None:
                lat, lon = self.centerline.latlon(s.x_m)
                row["lat_deg"] = lat
                row["lon_deg"] = lon

            rows.append(row)
            s = self.step(s, dt_s)

        return pd.DataFrame(rows)
```

## 9) `examples/run_a320.py`

```python
from __future__ import annotations

import numpy as np
from openap import aero

from approachsim.openap_adapter import build_aircraft_config
from approachsim.backends import OpenAPReferenceBackend
from approachsim.profiles import Centerline, build_simple_glidepath
from approachsim.profiles import ScalarProfile
from approachsim.weather import ConstantWind
from approachsim.dynamics import State
from approachsim.simulator import ApproachSimulator
from approachsim.openap_adapter import wrap_default
from approachsim.units import km_to_m


def build_centerline_example(intercept_distance_m: float) -> Centerline:
    # Dummy straight centerline for now.
    x = np.array([0.0, intercept_distance_m])
    lat = np.array([48.3538, 48.5000])
    lon = np.array([11.7861, 11.5000])
    return Centerline(x_m=x, lat_deg=lat, lon_deg=lon)


def build_speed_schedule_from_wrap(openap_objs) -> ScalarProfile:
    wrap = openap_objs.wrap
    v_des = wrap_default(wrap, "descent_const_vcas")
    v_final = wrap_default(wrap, "finalapp_vcas")
    v_land = wrap_default(wrap, "landing_speed")

    x_m = km_to_m(np.array([0.0, 8.0, 30.0, 60.0]))
    v_cas_mps = np.array([v_land, v_final, v_des, v_des], dtype=float)
    return ScalarProfile(x_m=x_m, y=v_cas_mps)


def main() -> None:
    cfg, o = build_aircraft_config("A320", payload_kg=12_000.0)
    perf = OpenAPReferenceBackend(cfg=cfg, openap=o)

    intercept_distance_m = 55_000.0
    threshold_elev_m = 450.0
    intercept_alt_m = 3_500.0

    altitude_profile = build_simple_glidepath(
        threshold_elevation_m=threshold_elev_m,
        intercept_distance_m=intercept_distance_m,
        intercept_altitude_m=intercept_alt_m,
        glide_deg=3.0,
    )
    speed_schedule = build_speed_schedule_from_wrap(o)
    centerline = build_centerline_example(intercept_distance_m)

    sim = ApproachSimulator(
        cfg=cfg,
        perf=perf,
        altitude_profile=altitude_profile,
        speed_schedule_cas=speed_schedule,
        centerline=centerline,
        weather=ConstantWind(alongtrack_mps=-10.0, delta_isa_K=0.0),  # 10 m/s headwind
    )

    # Start near the intercept.
    v0_cas = speed_schedule.value(intercept_distance_m)
    v0_tas = float(aero.cas2tas(v0_cas, intercept_alt_m))

    initial = State(
        t_s=0.0,
        x_m=intercept_distance_m,
        h_m=intercept_alt_m,
        v_tas_mps=v0_tas,
    )

    traj = sim.run(initial=initial, dt_s=1.0)
    print(traj.head())
    print(traj.tail())


if __name__ == "__main__":
    main()
```

## 10) What to verify first

OpenAP’s `FlightGenerator` can generate synthetic descent trajectories from the WRAP kinematic model and returns a DataFrame with columns including `t`, `h`, `s`, `v`, `vs`, `altitude`, `vertical_rate`, and `groundspeed`, so I would use it as a **sanity oracle** for regression tests, not as your simulator core. If your simulator’s no-wind descent behavior is wildly different from a corresponding WRAP trajectory for the same type, that is a sign that your mode gates or lags need tuning. ([OpenAP][5])

A simple regression fixture looks like this:

```python
from openap import FlightGenerator

gen = FlightGenerator("A320")
descent_df = gen.descent(dt=1, random=False, withcr=False)
print(descent_df[["t", "h", "v", "vs", "groundspeed"]].head())
```

## 11) My recommended implementation order

1. Get `build_aircraft_config("A320")` working.
2. Run `examples/run_a320.py` with zero wind.
3. Check that `v_cas_mps` moves from `descent_const_vcas` toward `finalapp_vcas` and then `landing_speed`.
4. Add headwind and tailwind tests and confirm that groundspeed changes immediately while CAS changes only through the lag model.
5. Replace the reference backend’s hard-coded `approach` and `final` flap mappings with your own fitted effective polar backend.


[1]: https://openap.dev/openap.html " OpenAP – The OpenAP Handbook"
[2]: https://openap.dev/aircraft_engine.html "1   Aircraft and engines – The OpenAP Handbook"
[3]: https://openap.dev/kinematic.html "3   Kinematic models – The OpenAP Handbook"
[4]: https://openap.dev/drag_thrust.html "2  ☯️ Drag and thrust – The OpenAP Handbook"
[5]: https://openap.dev/api/flightgenerator.html "20  ✈️ FlightGenerator – The OpenAP Handbook"
