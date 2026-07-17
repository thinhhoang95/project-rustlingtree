"""Region-scheduled exploration with replayable counts and RNG state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hailmary.config import LearningConfig
from hailmary.ids import canonical_data
from hailmary.learning.matching import AnchorContext, MatchSet
from hailmary.learning.rules import RuleAction


EXPLORATION_STATE_SCHEMA_VERSION = "hailmary.region_exploration.v1"
DEFAULT_COMMITMENT_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)
DEFAULT_PRESSURE_THRESHOLDS = (0.8, 1.2)
DEFAULT_ABS_ERROR_THRESHOLDS_S = (30.0, 90.0)


def _threshold_pair(
    values: Sequence[float],
    *,
    name: str,
) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two thresholds")
    lower, upper = (float(values[0]), float(values[1]))
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} thresholds must be finite and strictly increasing")
    return lower, upper


def _band(
    value: float, thresholds: tuple[float, float], labels: tuple[str, str, str]
) -> str:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("region features must be finite")
    if normalized < thresholds[0]:
        return labels[0]
    if normalized < thresholds[1]:
        return labels[1]
    return labels[2]


@dataclass(frozen=True, slots=True)
class RegionCell:
    commitment_band: str
    pressure_band: str
    abs_error_band: str

    def __post_init__(self) -> None:
        if self.commitment_band not in {"low", "medium", "high"}:
            raise ValueError("unknown commitment band")
        if self.pressure_band not in {"low", "nominal", "high"}:
            raise ValueError("unknown pressure band")
        if self.abs_error_band not in {"small", "medium", "large"}:
            raise ValueError("unknown absolute-error band")

    @property
    def key(self) -> tuple[str, str, str]:
        return (
            self.commitment_band,
            self.pressure_band,
            self.abs_error_band,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "commitment_band": self.commitment_band,
            "pressure_band": self.pressure_band,
            "abs_error_band": self.abs_error_band,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionCell":
        if not isinstance(payload, Mapping):
            raise TypeError("region cell payload must be a mapping")
        return cls(
            commitment_band=str(payload["commitment_band"]),
            pressure_band=str(payload["pressure_band"]),
            abs_error_band=str(payload["abs_error_band"]),
        )


@dataclass(frozen=True, slots=True)
class ExplorationDecision:
    cell: RegionCell
    action: RuleAction
    candidate: Any
    reason: str
    exploratory: bool
    visit_count: int
    action_experiment_count: int


class RegionExplorationScheduler:
    """Balance completed per-cell experiments before base exploration.

    ``choose`` records that a cell was visited, but deliberately does not
    claim that an experiment happened. The caller records a successful
    experiment only after its selected root action has committed by calling
    :meth:`record_experiment` with the returned decision.
    """

    def __init__(
        self,
        config: LearningConfig | None = None,
        *,
        seed: int | None = None,
        commitment_thresholds: Sequence[float] = DEFAULT_COMMITMENT_THRESHOLDS,
        pressure_thresholds: Sequence[float] = DEFAULT_PRESSURE_THRESHOLDS,
        abs_error_thresholds_s: Sequence[float] = DEFAULT_ABS_ERROR_THRESHOLDS_S,
    ) -> None:
        self.config = LearningConfig() if config is None else config
        if not isinstance(self.config, LearningConfig):
            raise TypeError("config must be LearningConfig")
        raw_seed = self.config.random_seed if seed is None else seed
        if isinstance(raw_seed, bool) or int(raw_seed) < 0:
            raise ValueError("exploration seed must be a non-negative integer")
        self.commitment_thresholds = _threshold_pair(
            commitment_thresholds,
            name="commitment",
        )
        self.pressure_thresholds = _threshold_pair(
            pressure_thresholds,
            name="pressure",
        )
        self.abs_error_thresholds_s = _threshold_pair(
            abs_error_thresholds_s,
            name="absolute-error",
        )
        self._rng = np.random.default_rng(int(raw_seed))
        self._visits: dict[RegionCell, int] = {}
        self._experiments: dict[tuple[RegionCell, RuleAction], int] = {}

    @staticmethod
    def _context(subject: AnchorContext | MatchSet) -> AnchorContext:
        if isinstance(subject, MatchSet):
            return subject.context
        if isinstance(subject, AnchorContext):
            return subject
        raise TypeError("exploration subject must be AnchorContext or MatchSet")

    def cell_for(self, subject: AnchorContext | MatchSet) -> RegionCell:
        context = self._context(subject)
        named = context.vector.named
        try:
            commitment = float(named["commitment_fraction"])
            pressure = float(named["pressure_ratio"])
        except KeyError as exc:
            raise ValueError(f"region vector is missing {exc.args[0]!r}") from exc
        if "abs_spacing_deviation_s" in named:
            abs_error = float(named["abs_spacing_deviation_s"])
        elif "spacing_deviation_s" in named:
            abs_error = abs(float(named["spacing_deviation_s"]))
        else:
            raise ValueError("region vector is missing absolute spacing error")
        return RegionCell(
            commitment_band=_band(
                commitment,
                self.commitment_thresholds,
                ("low", "medium", "high"),
            ),
            pressure_band=_band(
                pressure,
                self.pressure_thresholds,
                ("low", "nominal", "high"),
            ),
            abs_error_band=_band(
                abs_error,
                self.abs_error_thresholds_s,
                ("small", "medium", "large"),
            ),
        )

    def visit_count(self, cell: RegionCell) -> int:
        if not isinstance(cell, RegionCell):
            raise TypeError("cell must be RegionCell")
        return self._visits.get(cell, 0)

    def experiment_count(
        self,
        cell: RegionCell,
        action: RuleAction | Any,
    ) -> int:
        if not isinstance(cell, RegionCell):
            raise TypeError("cell must be RegionCell")
        normalized = RuleAction.from_candidate(action)
        return self._experiments.get((cell, normalized), 0)

    def choose(
        self,
        subject: AnchorContext | MatchSet,
        *,
        exploit_action: RuleAction | Any | None = None,
    ) -> ExplorationDecision:
        context = self._context(subject)
        actions = tuple(
            sorted(context.candidate_actions, key=lambda action: action.key)
        )
        if not actions:
            raise ValueError("exploration requires at least one feasible action")
        cell = self.cell_for(context)
        next_visit = self._visits.get(cell, 0) + 1
        self._visits[cell] = next_visit

        counts = {
            action: self._experiments.get((cell, action), 0) for action in actions
        }
        # A newly feasible action must receive one real experimental
        # opportunity even when this cell crossed its original coverage floor
        # before that action became available. Because counts are recorded
        # only after commit, a failed or skipped attempt remains undercovered.
        never_experimented = tuple(action for action in actions if counts[action] == 0)
        if never_experimented:
            pool = never_experimented
            index = 0 if len(pool) == 1 else int(self._rng.integers(len(pool)))
            selected = pool[index]
            reason = "coverage_floor"
            exploratory = True
        elif next_visit <= self.config.exploration_coverage_floor:
            minimum = min(counts.values())
            pool = tuple(action for action in actions if counts[action] == minimum)
            index = 0 if len(pool) == 1 else int(self._rng.integers(len(pool)))
            selected = pool[index]
            reason = "coverage_floor"
            exploratory = True
        elif (
            self.config.base_exploration_rate > 0.0
            and float(self._rng.random()) < self.config.base_exploration_rate
        ):
            index = 0 if len(actions) == 1 else int(self._rng.integers(len(actions)))
            selected = actions[index]
            reason = "base_exploration"
            exploratory = True
        else:
            preferred: RuleAction | None = None
            if exploit_action is not None:
                candidate_action = RuleAction.from_candidate(exploit_action)
                if candidate_action in counts:
                    preferred = candidate_action
            selected = actions[0] if preferred is None else preferred
            reason = "exploit" if preferred is not None else "deterministic_fallback"
            exploratory = False

        return ExplorationDecision(
            cell=cell,
            action=selected,
            candidate=context.candidate_for(selected),
            reason=reason,
            exploratory=exploratory,
            visit_count=next_visit,
            action_experiment_count=counts[selected],
        )

    def record_experiment(self, decision: ExplorationDecision) -> int:
        """Record one successfully committed decision exactly once."""

        if not isinstance(decision, ExplorationDecision):
            raise TypeError("decision must be ExplorationDecision")
        if self._visits.get(decision.cell, 0) != decision.visit_count:
            raise ValueError("cannot record a stale exploration decision")
        try:
            candidate_action = RuleAction.from_candidate(decision.candidate)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("exploration decision candidate is not canonical") from exc
        if candidate_action != decision.action:
            raise ValueError("exploration decision candidate/action mismatch")
        key = (decision.cell, decision.action)
        current = self._experiments.get(key, 0)
        if current != decision.action_experiment_count:
            raise ValueError("exploration decision is stale or already recorded")
        updated = current + 1
        self._experiments[key] = updated
        return updated

    @property
    def partition_configuration(self) -> Mapping[str, Any]:
        """Stable region boundaries that affect experimental allocation."""

        return MappingProxyType(
            {
                "commitment": self.commitment_thresholds,
                "pressure": self.pressure_thresholds,
                "abs_error_s": self.abs_error_thresholds_s,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        visits = [
            {"cell": cell.to_dict(), "count": count}
            for cell, count in sorted(
                self._visits.items(),
                key=lambda item: item[0].key,
            )
        ]
        experiments = [
            {
                "cell": cell.to_dict(),
                "action": action.to_dict(),
                "count": count,
            }
            for (cell, action), count in sorted(
                self._experiments.items(),
                key=lambda item: (*item[0][0].key, item[0][1].key),
            )
        ]
        return {
            "schema_version": EXPLORATION_STATE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "thresholds": canonical_data(self.partition_configuration),
            "visits": visits,
            "experiments": experiments,
            "rng_state": canonical_data(self._rng.bit_generator.state),
        }

    state_dict = to_dict

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegionExplorationScheduler":
        if not isinstance(payload, Mapping):
            raise TypeError("exploration state must be a mapping")
        if payload.get("schema_version") != EXPLORATION_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported exploration state schema")
        raw_config = payload.get("config")
        raw_thresholds = payload.get("thresholds")
        if not isinstance(raw_config, Mapping) or not isinstance(
            raw_thresholds, Mapping
        ):
            raise TypeError("exploration state config/thresholds must be mappings")
        scheduler = cls(
            LearningConfig(**dict(raw_config)),
            commitment_thresholds=raw_thresholds["commitment"],
            pressure_thresholds=raw_thresholds["pressure"],
            abs_error_thresholds_s=raw_thresholds["abs_error_s"],
        )

        for record in payload.get("visits", ()):
            if not isinstance(record, Mapping):
                raise TypeError("visit records must be mappings")
            cell = RegionCell.from_dict(record["cell"])
            count = record["count"]
            if type(count) is not int or count < 0:
                raise ValueError("visit counts must be non-negative integers")
            if cell in scheduler._visits:
                raise ValueError("duplicate visit cell in exploration state")
            scheduler._visits[cell] = count

        for record in payload.get("experiments", ()):
            if not isinstance(record, Mapping):
                raise TypeError("experiment records must be mappings")
            cell = RegionCell.from_dict(record["cell"])
            action = RuleAction.from_dict(record["action"])
            count = record["count"]
            if type(count) is not int or count < 0:
                raise ValueError("experiment counts must be non-negative integers")
            key = (cell, action)
            if key in scheduler._experiments:
                raise ValueError("duplicate action experiment in exploration state")
            scheduler._experiments[key] = count

        for cell, visits in scheduler._visits.items():
            experiments = sum(
                count
                for (experiment_cell, _), count in scheduler._experiments.items()
                if experiment_cell == cell
            )
            if experiments > visits:
                raise ValueError("region experiment counts cannot exceed visits")

        raw_rng_state = payload.get("rng_state")
        if not isinstance(raw_rng_state, Mapping):
            raise TypeError("rng_state must be a mapping")
        scheduler._rng.bit_generator.state = dict(raw_rng_state)
        return scheduler

    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "RegionExplorationScheduler":
        return cls.from_dict(payload)


__all__ = [
    "DEFAULT_ABS_ERROR_THRESHOLDS_S",
    "DEFAULT_COMMITMENT_THRESHOLDS",
    "DEFAULT_PRESSURE_THRESHOLDS",
    "EXPLORATION_STATE_SCHEMA_VERSION",
    "ExplorationDecision",
    "RegionCell",
    "RegionExplorationScheduler",
]
