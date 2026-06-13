from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_ppe_graph import _write_fixture
from vlm_ppe.cli import main


def test_cli_chosen_k_still_requires_vlm_for_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config_path = _write_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        main(["run-through-medoid", "--config", str(config_path), "--chosen-k", "2", "--run-id", "cli-run"])
