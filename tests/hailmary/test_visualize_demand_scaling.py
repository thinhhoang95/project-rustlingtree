from __future__ import annotations

import io
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo

from rich.console import Console

from hailmary.cli.visualize_demand_scaling import (
    DemandScalingInspector,
    WindowDeck,
    build_parser,
    run_interactive,
)
from hailmary.scenario import (
    ArrivalClusterKey,
    DemandWindow,
    TerminalEntryCorpus,
    TrafficScaleConfig,
)

from .test_phase01_traffic import _arrival, _builder


def test_cli_help_and_console_script_registration() -> None:
    help_text = build_parser().format_help()
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["scripts"]

    assert "hailmary-visualize-demand-scaling" in help_text
    assert (
        scripts["hailmary-visualize-demand-scaling"]
        == "hailmary.cli.visualize_demand_scaling:main"
    )


def _inspector(*, scale: float) -> DemandScalingInspector:
    key = ArrivalClusterKey("KATL", "RW18R", "C1")
    arrivals = tuple(
        _arrival(index, key, time_s)
        for index, time_s in enumerate((100.0, 500.0, 900.0, 1_300.0, 1_700.0, 2_100.0))
    )
    corpus = TerminalEntryCorpus(
        dataset_id="phase01-test",
        arrivals=arrivals,
        rejection_counts=(),
        airport="KATL",
    )
    return DemandScalingInspector(
        corpus,
        _builder(arrivals),
        windows=(DemandWindow(0.0, 3_600.0), DemandWindow(1_200.0, 4_800.0)),
        scale_config=TrafficScaleConfig(
            global_scale=scale,
            master_seed=23,
            replicate=2,
        ),
        timezone=ZoneInfo("UTC"),
        verbose=True,
    )


def _console() -> tuple[Console, io.StringIO]:
    output = io.StringIO()
    return (
        Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=240,
        ),
        output,
    )


def test_rendered_upscale_marks_synthetic_flights_and_orders_releases() -> None:
    inspector = _inspector(scale=1.5)
    sample = inspector.sample(DemandWindow(0.0, 3_600.0))
    console, output = _console()

    inspector.render(console, sample)
    rendered = output.getvalue()

    assert sample.original_count == 6
    assert sample.new_count == 9
    assert sample.synthetic_count == 3
    assert "Original count: 6" in rendered
    assert "New count: 9" in rendered
    assert "synthetic arrival" in rendered
    assert "Donor Flight" in rendered
    inspector.verbose = False
    plain_console, plain_output = _console()
    inspector.render(plain_console, sample)
    plain_rendered = plain_output.getvalue()
    ordered = sorted(
        sample.scenario.definition.flights,
        key=lambda item: (item.release_time_s, item.flight_id),
    )
    positions = [plain_rendered.index(flight.flight_id) for flight in ordered]
    assert positions == sorted(positions)


def test_rendered_downscale_lists_removed_observed_flights() -> None:
    inspector = _inspector(scale=0.5)
    sample = inspector.sample(DemandWindow(0.0, 3_600.0))
    console, output = _console()

    inspector.render(console, sample)
    rendered = output.getvalue()

    assert sample.original_count == 6
    assert sample.new_count == 3
    assert sample.synthetic_count == 0
    assert len(sample.removed_arrivals) == 3
    assert "Removed Flights (Seeded Thinning)" in rendered
    assert all(item.flight_id in rendered for item in sample.removed_arrivals)


def test_window_deck_is_seeded_and_does_not_repeat_within_a_pass() -> None:
    windows = tuple(
        DemandWindow(float(index * 1_200), float(index * 1_200 + 3_600))
        for index in range(4)
    )
    first = WindowDeck(windows, seed=7)
    second = WindowDeck(windows, seed=7)

    first_pass = tuple(first.draw() for _ in windows)
    second_pass = tuple(second.draw() for _ in windows)

    assert first_pass == second_pass
    assert len(set(first_pass)) == len(windows)


def test_interactive_enter_renders_another_sample_and_q_exits() -> None:
    inspector = _inspector(scale=1.5)
    console, output = _console()
    choices = iter(("1", "", "q"))
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        return next(choices)

    run_interactive(inspector, console, read=read)

    rendered = output.getvalue()
    assert "Demand Scaling Inspector" in rendered
    assert rendered.count("Demand Scaling Sample") == 2
    assert rendered.count("Flight List") == 2
    assert any("Enter" in prompt and "another sample" in prompt for prompt in prompts)


def test_initial_exact_window_enters_the_same_repeat_loop() -> None:
    inspector = _inspector(scale=1.5)
    initial = inspector.sample(DemandWindow(0.0, 3_600.0))
    console, output = _console()
    choices = iter(("", "q"))

    run_interactive(
        inspector,
        console,
        read=lambda _prompt: next(choices),
        initial_sample=initial,
    )

    rendered = output.getvalue()
    assert rendered.count("Demand Scaling Sample") == 2
    assert rendered.count("Flight List") == 2


def test_exact_window_start_accepts_local_iso_time() -> None:
    inspector = _inspector(scale=1.0)

    window = inspector.parse_window_start("1970-01-01 00:20")

    assert window == DemandWindow(1_200.0, 4_800.0)
