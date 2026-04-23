#!/usr/bin/env python3
# Usage example: python -m scenario.adip_resources.download_kdfw_charts data/adip/kdfw_adip_resources.json --output-dir data/adip/charts
"""Download KDFW ADIP STAR and IAP chart PDFs one by one."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/adip/kdfw_adip_resources.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/adip/charts"
USER_AGENT = "Mozilla/5.0 (compatible; KDFWChartDownloader/1.0)"
CHUNK_SIZE = 1024 * 64
REQUEST_TIMEOUT_SECONDS = 60
SECTION_ORDER = ("star", "iap")


@dataclass(slots=True)
class ChartEntry:
    """A single downloadable chart entry."""

    section: str
    label: str
    url: str
    filename: str
    destination: Path


@dataclass(slots=True)
class DownloadSummary:
    """Aggregate result from a download run."""

    manifest_path: Path
    output_dir: Path
    total: int
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)


def load_manifest(manifest_path: Path | str) -> dict[str, list[dict[str, str]]]:
    """Load and validate the chart manifest JSON."""

    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a JSON object.")

    manifest: dict[str, list[dict[str, str]]] = {}
    for section in SECTION_ORDER:
        entries = data.get(section, [])
        if not isinstance(entries, list):
            raise ValueError(f"Manifest section '{section}' must be a list.")
        validated_entries: list[dict[str, str]] = []
        for item in entries:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(
                    f"Manifest section '{section}' entries must be single-key objects."
                )
            label, url = next(iter(item.items()))
            if not isinstance(label, str) or not isinstance(url, str):
                raise ValueError(
                    f"Manifest section '{section}' entries must map strings to strings."
                )
            validated_entries.append({label: url})
        manifest[section] = validated_entries

    return manifest


def iter_chart_entries(
    manifest: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> Iterable[ChartEntry]:
    """Yield flattened chart entries in STAR then IAP order."""

    for section in SECTION_ORDER:
        for item in manifest.get(section, []):
            label, url = next(iter(item.items()))
            filename = unquote(Path(urlparse(url).path).name)
            if not filename:
                raise ValueError(f"Could not determine filename from chart URL: {url}")
            yield ChartEntry(
                section=section,
                label=label,
                url=url,
                filename=filename,
                destination=output_dir / filename,
            )


def _download_to_path(url: str, destination: Path, progress: Progress, task_id: int) -> None:
    """Stream one URL to disk with a temporary file and a progress task."""

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
        },
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_length = None
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw_length = headers.get("Content-Length")
            if raw_length and raw_length.isdigit():
                content_length = int(raw_length)

        if content_length is not None:
            progress.update(task_id, total=content_length)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(destination.parent),
            prefix=f".{destination.stem}.",
            suffix=".part",
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            written = 0
            try:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                    written += len(chunk)
                    progress.update(task_id, completed=written)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    tmp_path.replace(destination)


def download_charts(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    console: Console | None = None,
) -> DownloadSummary:
    """Download all charts from the manifest into the output directory."""

    console = console or Console()
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    entries = list(iter_chart_entries(manifest, output_dir))
    summary = DownloadSummary(
        manifest_path=manifest_path,
        output_dir=output_dir,
        total=len(entries),
    )

    console.print(
        Panel.fit(
            f"[bold]Manifest:[/bold] {manifest_path}\n"
            f"[bold]Output:[/bold] {output_dir}\n"
            f"[bold]Charts:[/bold] {summary.total} total",
            title="KDFW ADIP Download Manager",
        )
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        overall_task = progress.add_task("[cyan]Charts", total=summary.total)
        for entry in entries:
            if entry.destination.exists():
                summary.skipped += 1
                progress.console.print(
                    f"[yellow]Skip[/yellow] {entry.filename} already exists"
                )
                progress.advance(overall_task)
                continue

            task_id = progress.add_task(
                f"[green]{entry.section.upper()}[/green] {entry.label}",
                total=None,
            )
            try:
                _download_to_path(entry.url, entry.destination, progress, task_id)
                summary.downloaded += 1
                progress.console.print(
                    f"[green]Saved[/green] {entry.filename} -> {entry.destination}"
                )
            except (HTTPError, URLError, OSError, ValueError) as exc:
                summary.failed += 1
                summary.failures.append(f"{entry.filename}: {exc}")
                progress.console.print(f"[red]Failed[/red] {entry.filename}: {exc}")
            finally:
                progress.remove_task(task_id)
                progress.advance(overall_task)

    _render_summary(console, summary)
    return summary


def _render_summary(console: Console, summary: DownloadSummary) -> None:
    table = Table(title="Download Summary", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Manifest", str(summary.manifest_path))
    table.add_row("Output directory", str(summary.output_dir))
    table.add_row("Total charts", str(summary.total))
    table.add_row("Downloaded", str(summary.downloaded))
    table.add_row("Skipped", str(summary.skipped))
    table.add_row("Failed", str(summary.failed))
    console.print(table)

    if summary.failures:
        failure_table = Table(title="Failures", expand=False)
        failure_table.add_column("Chart", style="bold red")
        failure_table.add_column("Error")
        for failure in summary.failures:
            chart, _, message = failure.partition(": ")
            failure_table.add_row(chart, message)
        console.print(failure_table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download STAR and IAP chart PDFs from an ADIP manifest JSON."
    )
    parser.add_argument(
        "manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        nargs="?",
        help=f"Path to the chart manifest JSON (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where PDFs will be saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        download_charts(args.manifest, args.output_dir)
    except Exception as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
