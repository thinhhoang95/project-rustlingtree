"""Compatibility exports for longitudinal profile helpers."""

from .nlp_colloc.profiles import ConstraintEnvelope, ScalarProfile, build_speed_schedule_from_wrap

__all__ = [
    "ConstraintEnvelope",
    "ScalarProfile",
    "build_speed_schedule_from_wrap",
]
