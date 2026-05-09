from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from .envelope_viz import build_bichannel_result, plot_cross_check

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "x_check_simap_adsb.py"
_SCRIPT_SPEC = spec_from_file_location("_x_check_simap_adsb_script", _SCRIPT_PATH)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise ImportError(f"Unable to load script helpers from {_SCRIPT_PATH}")

sys.modules.setdefault("x_check_simap_adsb", sys.modules[__name__])
_SCRIPT_MODULE = module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = _SCRIPT_MODULE
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)

derive_speed_mps = _SCRIPT_MODULE.derive_speed_mps
load_simap_payload = _SCRIPT_MODULE.load_simap_payload
parse_flight_key = _SCRIPT_MODULE.parse_flight_key
sample_trajectory = _SCRIPT_MODULE.sample_trajectory
trajectory_from_simap_payload = _SCRIPT_MODULE.trajectory_from_simap_payload

__all__ = [
    "build_bichannel_result",
    "derive_speed_mps",
    "load_simap_payload",
    "parse_flight_key",
    "plot_cross_check",
    "sample_trajectory",
    "trajectory_from_simap_payload",
]
