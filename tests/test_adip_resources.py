from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from scenario.adip_resources.download_kdfw_charts import (
    download_charts,
    iter_chart_entries,
    load_manifest,
)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)
        self.headers = {}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_iter_chart_entries_keeps_section_order(tmp_path: Path) -> None:
    manifest = {
        "star": [{"STAR_ONE": "https://example.com/one.PDF"}],
        "iap": [{"IAP_ONE": "https://example.com/two.PDF"}],
    }

    entries = list(iter_chart_entries(manifest, tmp_path))

    assert [entry.section for entry in entries] == ["star", "iap"]
    assert entries[0].destination == tmp_path / "one.PDF"
    assert entries[1].destination == tmp_path / "two.PDF"


def test_download_charts_skips_existing_and_downloads_missing(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "manifest.json"
    output_dir = tmp_path / "charts"
    output_dir.mkdir()
    (output_dir / "one.PDF").write_bytes(b"existing")

    manifest = {
        "star": [{"STAR_ONE": "https://example.com/one.PDF"}],
        "iap": [{"IAP_TWO": "https://example.com/two.PDF"}],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_urlopen(request, timeout=60):  # noqa: ARG001
        url = request.full_url
        if url.endswith("two.PDF"):
            return _FakeResponse(b"downloaded")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        "scenario.adip_resources.download_kdfw_charts.urlopen",
        fake_urlopen,
    )

    console = Console(file=io.StringIO(), force_terminal=True, width=100)
    summary = download_charts(manifest_path, output_dir, console=console)

    assert summary.total == 2
    assert summary.skipped == 1
    assert summary.downloaded == 1
    assert summary.failed == 0
    assert (output_dir / "one.PDF").read_bytes() == b"existing"
    assert (output_dir / "two.PDF").read_bytes() == b"downloaded"


def test_load_manifest_validates_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    try:
        load_manifest(manifest_path)
    except ValueError as exc:
        assert "Manifest root" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

