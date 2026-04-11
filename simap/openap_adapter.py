from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from openap import prop
from openap.drag import Drag
from openap.kinematic import WRAP
from openap.thrust import Thrust


@dataclass(frozen=True)
class OpenAPAircraftData:
    typecode: str
    engine_name: str
    wing_area_m2: float
    clean_cd0: float
    clean_k: float
    vmo_kts: float
    mmo: float
    oew_kg: float
    mlw_kg: float
    mtow_kg: float


@dataclass(frozen=True)
class OpenAPObjects:
    aircraft: dict[str, Any]
    engine: dict[str, Any]
    wrap: WRAP
    drag: Drag
    thrust: Thrust


def load_openap(typecode: str, engine_name: str | None = None) -> OpenAPObjects:
    aircraft = prop.aircraft(typecode)
    selected_engine = engine_name or aircraft["engine"]["default"]
    engine = prop.engine(selected_engine)
    return OpenAPObjects(
        aircraft=aircraft,
        engine=engine,
        wrap=WRAP(ac=typecode),
        drag=Drag(ac=typecode),
        thrust=Thrust(ac=typecode, eng=selected_engine),
    )


def extract_aircraft_data(openap: OpenAPObjects) -> OpenAPAircraftData:
    aircraft = openap.aircraft
    return OpenAPAircraftData(
        typecode=str(aircraft["aircraft"]),
        engine_name=str(aircraft["engine"]["default"]),
        wing_area_m2=float(aircraft["wing"]["area"]),
        clean_cd0=float(aircraft["drag"]["cd0"]),
        clean_k=float(aircraft["drag"]["k"]),
        vmo_kts=float(aircraft["vmo"]),
        mmo=float(aircraft["mmo"]),
        oew_kg=float(aircraft["oew"]),
        mlw_kg=float(aircraft["mlw"]),
        mtow_kg=float(aircraft["mtow"]),
    )


def wrap_default(wrap: WRAP, method_name: str) -> float:
    return float(getattr(wrap, method_name)()["default"])


def wrap_sample(wrap: WRAP, method_name: str, rng: np.random.Generator) -> float:
    params = getattr(wrap, method_name)()
    model = getattr(stats, params["statmodel"])(*params["statmodel_params"])
    sample = model.rvs(random_state=rng)
    return float(np.clip(sample, params["minimum"], params["maximum"]))
