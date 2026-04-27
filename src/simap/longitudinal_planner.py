"""Compatibility wrapper for the renamed coupled descent planner module."""

from .coupled_descent_planner import *  # noqa: F401,F403
from .smooth_idle_planner import plan_smooth_idle_descent  # noqa: F401
