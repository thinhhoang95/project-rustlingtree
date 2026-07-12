"""Domain errors raised by the Hailmary simulator."""

from __future__ import annotations


class HailmaryError(Exception):
    """Base class for errors with simulator-domain meaning."""


class ArtifactValidationError(HailmaryError, ValueError):
    """An offline artifact violates the executable contract."""


class ConfigurationError(HailmaryError, ValueError):
    """Configuration values are inconsistent or unsupported."""


class InfeasibleActionError(HailmaryError, ValueError):
    """An action cannot be realized from the current state."""


class StaleActionError(InfeasibleActionError):
    """An action was computed for a different state or decision epoch."""


class SimulationError(HailmaryError, RuntimeError):
    """The event engine cannot safely advance the scenario."""


class CorrelationGateError(HailmaryError, ValueError):
    """A training scenario failed the registered feature-correlation gate."""


class NoFeasibleClusterError(HailmaryError, ValueError):
    """No usable clustering configuration or fallback could be produced."""
