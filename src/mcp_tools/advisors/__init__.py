"""Advisory tools for scenario-manager workflows."""

from .feasibility import FeasibilityAdvisor
from .models import (
    AdvisoryFlight,
    BaseAdvisory,
    FeasibilityAdvisory,
    SpeedControlAdvisory,
    VectoringAdvisory,
)
from .profile import ArrivalProfile, PlannedProfile, ProfilePlanner, extend_reference_path
from .speed_control import SpeedControlAdvisor
from .vectoring import VectoringAdvisor

__all__ = [
    "AdvisoryFlight",
    "ArrivalProfile",
    "BaseAdvisory",
    "FeasibilityAdvisor",
    "FeasibilityAdvisory",
    "PlannedProfile",
    "ProfilePlanner",
    "SpeedControlAdvisor",
    "SpeedControlAdvisory",
    "VectoringAdvisor",
    "VectoringAdvisory",
    "extend_reference_path",
]
