from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stem",
    ("01_demand_and_scaling", "02_route_graph_and_pairing"),
)
def test_example_output_matches_checked_result(stem: str, tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    example_dir = repository / "examples" / "hailmary_phase01"
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    completed = subprocess.run(
        [sys.executable, str(example_dir / f"{stem}.py")],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    expected = json.loads(
        (example_dir / "results" / f"{stem}.json").read_text(encoding="utf-8")
    )
    assert json.loads(completed.stdout) == expected
