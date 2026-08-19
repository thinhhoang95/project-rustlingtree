"""Interactively inspect deterministic Hailmary demand-scaling realizations."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
import random
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hailmary.scenario import (
    DemandWindow,
    DemandWindowConfig,
    ObservedArrival,
    TerminalEntryCorpus,
    TrafficScaleConfig,
    TrafficScenario,
    TrafficScenarioBuilder,
    iter_demand_windows,
)
from hailmary.templates import TemplateStore

ChoiceReader = Callable[[str], str]


def _timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"unknown IANA timezone: {value!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hailmary-visualize-demand-scaling",
        description=(
            "Interactively materialize one independent demand window and inspect "
            "its observed, synthetic, and thinned arrivals."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus/traffic_corpus.json"),
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("data/artifacts/hailmary/corpus/hailmary_templates.json"),
    )
    parser.add_argument(
        "--start",
        type=float,
        help="first candidate window start as a Unix timestamp (defaults to corpus range)",
    )
    parser.add_argument(
        "--stop",
        type=float,
        help="exclusive stop for candidate window starts (defaults to corpus range)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--timezone",
        type=_timezone,
        default=ZoneInfo("UTC"),
        help="IANA timezone used for display and typed window starts (default: UTC)",
    )
    parser.add_argument(
        "--window-start",
        help=(
            "start inspection from this window; accepts a Unix timestamp or an ISO "
            "date/time such as '2026-04-01 09:00'"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show donor flight and intensity-scope columns",
    )
    return parser


@dataclass(frozen=True, slots=True)
class ScalingSample:
    """One production scaling realization plus its observed-source comparison."""

    scenario: TrafficScenario
    original_arrivals: tuple[ObservedArrival, ...]
    removed_arrivals: tuple[ObservedArrival, ...]

    @property
    def original_count(self) -> int:
        return len(self.original_arrivals)

    @property
    def new_count(self) -> int:
        return len(self.scenario.definition.flights)

    @property
    def synthetic_count(self) -> int:
        return sum(
            bool(flight.metadata_dict.get("synthetic"))
            for flight in self.scenario.definition.flights
        )


class WindowDeck:
    """Seeded, no-repeat window selection with reshuffling after one full pass."""

    def __init__(self, windows: Sequence[DemandWindow], *, seed: int) -> None:
        self._windows = tuple(windows)
        if not self._windows:
            raise ValueError("window deck requires at least one window")
        self._rng = random.Random(seed)
        self._remaining: list[DemandWindow] = []

    def draw(self) -> DemandWindow:
        if not self._remaining:
            self._remaining = list(self._windows)
            self._rng.shuffle(self._remaining)
        return self._remaining.pop()


class DemandScalingInspector:
    """Build and render independently scaled windows from one offline corpus."""

    def __init__(
        self,
        corpus: TerminalEntryCorpus,
        builder: TrafficScenarioBuilder,
        *,
        windows: Sequence[DemandWindow],
        scale_config: TrafficScaleConfig,
        timezone: ZoneInfo,
        verbose: bool = False,
    ) -> None:
        self.corpus = corpus
        self.builder = builder
        self.scale_config = scale_config
        self.timezone = timezone
        self.verbose = bool(verbose)
        self.windows = _nonempty_windows(corpus.arrivals, windows)
        if not self.windows:
            raise ValueError("configured range contains no observed traffic windows")
        self.deck = WindowDeck(self.windows, seed=scale_config.master_seed)

    def sample(self, window: DemandWindow) -> ScalingSample:
        original = tuple(
            arrival
            for arrival in self.corpus.arrivals
            if window.contains(arrival.terminal_entry_time_s)
        )
        if not original:
            raise ValueError("selected window contains no observed arrivals")
        scenario = self.builder.build_scenario(
            window,
            scale_config=self.scale_config,
        )
        retained_ids = {flight.flight_id for flight in scenario.definition.flights}
        removed = tuple(
            arrival for arrival in original if arrival.flight_id not in retained_ids
        )
        return ScalingSample(
            scenario=scenario,
            original_arrivals=original,
            removed_arrivals=removed,
        )

    def draw(self) -> ScalingSample:
        return self.sample(self.deck.draw())

    def parse_window_start(self, value: str) -> DemandWindow:
        start = _parse_time(value, self.timezone)
        configured_start = min(window.start_s for window in self.windows)
        configured_stop = max(window.start_s for window in self.windows)
        if start < configured_start or start > configured_stop:
            raise ValueError("window start lies outside the configured candidate range")
        return DemandWindow(start, start + DemandWindowConfig().width_s)

    def render(self, console: Console, sample: ScalingSample) -> None:
        _render_sample(
            console,
            sample,
            timezone=self.timezone,
            verbose=self.verbose,
            scale_config=self.scale_config,
        )


def _default_bounds(
    arrivals: Sequence[ObservedArrival],
    *,
    start_s: float | None,
    stop_s: float | None,
) -> tuple[float, float]:
    if not arrivals:
        raise ValueError("cannot derive demand windows from an empty corpus")
    stride = DemandWindowConfig().stride_s
    earliest = min(item.terminal_entry_time_s for item in arrivals)
    latest = max(item.terminal_entry_time_s for item in arrivals)
    start = (
        math.floor(earliest / stride) * stride if start_s is None else float(start_s)
    )
    stop = (
        (math.floor(latest / stride) + 1) * stride if stop_s is None else float(stop_s)
    )
    return start, stop


def _nonempty_windows(
    arrivals: Sequence[ObservedArrival],
    windows: Sequence[DemandWindow],
) -> tuple[DemandWindow, ...]:
    times = sorted(item.terminal_entry_time_s for item in arrivals)
    return tuple(
        window
        for window in windows
        if (index := bisect_left(times, window.start_s)) < len(times)
        and times[index] < window.end_s
    )


def _parse_time(value: str, timezone: ZoneInfo) -> float:
    token = str(value).strip()
    if not token:
        raise ValueError("window start cannot be empty")
    try:
        timestamp = float(token)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(token)
        except ValueError as exc:
            raise ValueError(
                "window start must be a Unix timestamp or ISO date/time"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone)
        timestamp = parsed.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("window start must be finite")
    return timestamp


def _window_label(window: DemandWindow, timezone: ZoneInfo) -> str:
    start = datetime.fromtimestamp(window.start_s, tz=UTC).astimezone(timezone)
    end = datetime.fromtimestamp(window.end_s, tz=UTC).astimezone(timezone)
    zone = getattr(timezone, "key", str(timezone))
    if start.date() == end.date():
        return f"{start:%Y-%m-%d %H:%M}–{end:%H:%M} {zone}"
    return f"{start:%Y-%m-%d %H:%M}–{end:%Y-%m-%d %H:%M} {zone}"


def _release_label(time_s: float, timezone: ZoneInfo) -> str:
    return (
        datetime.fromtimestamp(time_s, tz=UTC).astimezone(timezone).strftime("%H:%M:%S")
    )


def _render_sample(
    console: Console,
    sample: ScalingSample,
    *,
    timezone: ZoneInfo,
    verbose: bool,
    scale_config: TrafficScaleConfig,
) -> None:
    scenario = sample.scenario
    added = sample.synthetic_count
    removed = len(sample.removed_arrivals)
    summary = "\n".join(
        (
            f"[bold]Selected window:[/bold] {_window_label(scenario.window, timezone)}",
            (
                f"[bold]Scale:[/bold] {scale_config.global_scale:g}  ·  "
                f"[bold]Seed:[/bold] {scale_config.master_seed}  ·  "
                f"[bold]Replicate:[/bold] {scale_config.replicate}"
            ),
            (
                f"[bold]Original count:[/bold] {sample.original_count:,}  ·  "
                f"[bold]New count:[/bold] {sample.new_count:,}  ·  "
                f"[green]Added: {added:,}[/green]  ·  "
                f"[red]Removed: {removed:,}[/red]"
            ),
        )
    )
    console.print(Panel(summary, title="Demand Scaling Sample", border_style="cyan"))

    observed = {item.key: item.count for item in scenario.observed_cluster_counts}
    target = {item.key: item.count for item in scenario.target_cluster_counts}
    counts = Table(title="Runway / Cluster Counts", box=box.ROUNDED)
    counts.add_column("Runway", style="cyan")
    counts.add_column("Cluster")
    counts.add_column("Original", justify="right")
    counts.add_column("Target", justify="right")
    counts.add_column("Difference", justify="right")
    for key in sorted(observed):
        if observed[key] == 0 and target[key] == 0:
            continue
        difference = target[key] - observed[key]
        difference_text = f"{difference:+d}" if difference else "0"
        style = "green" if difference > 0 else "red" if difference < 0 else None
        counts.add_row(
            key.runway,
            key.cluster,
            str(observed[key]),
            str(target[key]),
            difference_text,
            style=style,
        )
    console.print(counts)

    flights = Table(title="Flight List", box=box.ROUNDED, row_styles=("", "dim"))
    flights.add_column("#", justify="right", no_wrap=True)
    flights.add_column("Flight ID", overflow="fold")
    flights.add_column("Release", no_wrap=True)
    flights.add_column("Runway", no_wrap=True)
    flights.add_column("Cluster", no_wrap=True)
    flights.add_column("Status", no_wrap=True)
    if verbose:
        flights.add_column("Donor Flight", overflow="fold")
        flights.add_column("Intensity", no_wrap=True)
    ordered = sorted(
        scenario.definition.flights,
        key=lambda item: (item.release_time_s, item.flight_id),
    )
    for index, flight in enumerate(ordered, start=1):
        metadata = flight.metadata_dict
        synthetic = bool(metadata.get("synthetic"))
        cells = [
            f"{index}*" if synthetic else str(index),
            flight.flight_id,
            _release_label(flight.release_time_s, timezone),
            flight.runway,
            flight.cluster_id.rsplit(":", 1)[-1],
            "[yellow]synthetic[/yellow]" if synthetic else "observed",
        ]
        if verbose:
            cells.extend(
                (
                    str(metadata.get("donor_flight_id") or "—"),
                    str(metadata.get("intensity_scope") or "—"),
                )
            )
        flights.add_row(*cells)
    console.print(flights)
    console.print("[yellow]*[/yellow] synthetic arrival")

    if sample.removed_arrivals:
        removed_table = Table(
            title="Removed Flights (Seeded Thinning)",
            box=box.ROUNDED,
        )
        removed_table.add_column("Flight ID")
        removed_table.add_column("Release", no_wrap=True)
        removed_table.add_column("Runway", no_wrap=True)
        removed_table.add_column("Cluster", no_wrap=True)
        for arrival in sorted(
            sample.removed_arrivals,
            key=lambda item: (item.terminal_entry_time_s, item.flight_id),
        ):
            removed_table.add_row(
                arrival.flight_id,
                _release_label(arrival.terminal_entry_time_s, timezone),
                arrival.key.runway,
                arrival.key.cluster,
            )
        console.print(removed_table)


def _load_inspector(args: argparse.Namespace) -> DemandScalingInspector:
    corpus = TerminalEntryCorpus.read(args.corpus)
    store = TemplateStore.read(args.templates)
    templates = {
        f"{template.airport_id}:{template.runway_id}:{template.cluster_id}": template
        for template in store.templates
    }
    builder = TrafficScenarioBuilder(
        corpus.arrivals,
        templates_by_cluster=templates,
        rejection_counts=dict(corpus.rejection_counts),
    )
    start, stop = _default_bounds(
        corpus.arrivals,
        start_s=args.start,
        stop_s=args.stop,
    )
    windows = iter_demand_windows(start, stop, config=DemandWindowConfig())
    return DemandScalingInspector(
        corpus,
        builder,
        windows=windows,
        scale_config=TrafficScaleConfig(
            global_scale=args.scale,
            replicate=args.replicate,
            master_seed=args.seed,
        ),
        timezone=args.timezone,
        verbose=args.verbose,
    )


def run_interactive(
    inspector: DemandScalingInspector,
    console: Console,
    *,
    read: ChoiceReader | None = None,
    initial_sample: ScalingSample | None = None,
) -> None:
    reader = console.input if read is None else read
    menu = Panel.fit(
        "[bold cyan]1[/bold cyan]  Generate a training traffic sample\n"
        "[bold cyan]2[/bold cyan]  Select an exact window\n"
        "[bold cyan]Q[/bold cyan]  Quit",
        title="Demand Scaling Inspector",
    )

    def browse(first_sample: ScalingSample) -> bool:
        sample = first_sample
        while True:
            inspector.render(console, sample)
            next_choice = (
                reader(
                    "\nPress [bold cyan]Enter[/bold cyan] for another sample, "
                    "[bold cyan]M[/bold cyan] for menu, or "
                    "[bold cyan]Q[/bold cyan] to quit: "
                )
                .strip()
                .lower()
            )
            if next_choice in {"q", "quit", "exit"}:
                return True
            if next_choice in {"m", "menu"}:
                return False
            sample = inspector.draw()

    if initial_sample is not None:
        if browse(initial_sample):
            return
    console.print(menu)

    while True:
        choice = reader("\n[bold]> [/bold]").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return
        if choice == "1":
            if browse(inspector.draw()):
                return
            console.print(menu)
            continue
        if choice == "2":
            raw_start = reader("Window start (Unix timestamp or YYYY-MM-DD HH:MM): ")
            try:
                window = inspector.parse_window_start(raw_start)
                if browse(inspector.sample(window)):
                    return
                console.print(menu)
            except ValueError as exc:
                console.print(f"[red]Cannot build window:[/red] {exc}")
            continue
        console.print("[yellow]Choose 1, 2, or Q.[/yellow]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        inspector = _load_inspector(args)
        if args.window_start is not None:
            window = inspector.parse_window_start(args.window_start)
            run_interactive(
                inspector,
                console,
                initial_sample=inspector.sample(window),
            )
        else:
            run_interactive(inspector, console)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DemandScalingInspector",
    "ScalingSample",
    "WindowDeck",
    "build_parser",
    "main",
    "run_interactive",
]
