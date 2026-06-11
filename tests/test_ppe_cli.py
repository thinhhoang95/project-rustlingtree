from __future__ import annotations

from pathlib import Path

from tests.test_ppe_graph import _write_fixture
from vlm_ppe.cli import main


def test_cli_smoke_runs_offline_chosen_k(tmp_path: Path, capsys) -> None:
    config_path = _write_fixture(tmp_path)

    exit_code = main(["run-through-medoid", "--config", str(config_path), "--chosen-k", "2", "--run-id", "cli-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"status": "complete"' in captured.out
