#!/usr/bin/env python3
#Usage example: python -m scenario.adip_resources.convert_adip_charts_to_jpg data/adip/charts --output-dir src/scenario/adip_resources/img

"""Convert ADIP chart PDFs into JPEG images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHARTS_DIR = PROJECT_ROOT / "data/adip/charts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src/scenario/adip_resources/img"
DEFAULT_DPI = 300
DEFAULT_JPEG_QUALITY = 95


@dataclass(slots=True)
class ConversionSummary:
    """Aggregate result from a conversion run."""

    charts_dir: Path
    output_dir: Path
    total_pdfs: int
    converted_pages: int = 0
    skipped_pages: int = 0
    failed_pdfs: int = 0
    failures: list[str] = field(default_factory=list)


def iter_pdf_files(charts_dir: Path) -> list[Path]:
    """Return chart PDFs in a stable order."""

    if not charts_dir.exists():
        raise FileNotFoundError(f"Charts directory does not exist: {charts_dir}")
    if not charts_dir.is_dir():
        raise NotADirectoryError(f"Charts path is not a directory: {charts_dir}")

    pdfs = [
        path
        for path in sorted(charts_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    if not pdfs:
        raise FileNotFoundError(f"No PDF charts found in: {charts_dir}")
    return pdfs


def _page_output_path(output_dir: Path, pdf_path: Path, page_number: int, page_count: int) -> Path:
    """Build the target JPEG path for a given PDF page."""

    if page_count == 1:
        return output_dir / f"{pdf_path.stem}.jpg"
    return output_dir / f"{pdf_path.stem}_page{page_number:03d}.jpg"


def _save_pdf_as_jpegs(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int,
    quality: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Convert one PDF into one or more JPEG files."""

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    converted = 0
    skipped = 0

    for index, page in enumerate(pages, start=1):
        destination = _page_output_path(output_dir, pdf_path, index, len(pages))
        if destination.exists() and not overwrite:
            skipped += 1
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        image = page.convert("RGB")
        image.save(destination, format="JPEG", quality=quality, optimize=True)
        converted += 1

    return converted, skipped


def convert_charts(
    charts_dir: Path | str = DEFAULT_CHARTS_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    dpi: int = DEFAULT_DPI,
    quality: int = DEFAULT_JPEG_QUALITY,
    overwrite: bool = False,
    console: Console | None = None,
) -> ConversionSummary:
    """Convert all chart PDFs in a directory to JPEGs."""

    console = console or Console()
    charts_dir = Path(charts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = iter_pdf_files(charts_dir)
    summary = ConversionSummary(
        charts_dir=charts_dir,
        output_dir=output_dir,
        total_pdfs=len(pdf_files),
    )

    console.print(
        Panel.fit(
            f"[bold]Charts:[/bold] {charts_dir}\n"
            f"[bold]Output:[/bold] {output_dir}\n"
            f"[bold]PDFs:[/bold] {summary.total_pdfs} total\n"
            f"[bold]DPI:[/bold] {dpi}  [bold]Quality:[/bold] {quality}",
            title="ADIP Chart Converter",
        )
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    with progress:
        overall_task = progress.add_task("[cyan]PDF charts", total=summary.total_pdfs)
        for pdf_path in pdf_files:
            progress.console.print(f"[cyan]Converting[/cyan] {pdf_path.name}")
            try:
                converted, skipped = _save_pdf_as_jpegs(
                    pdf_path,
                    output_dir,
                    dpi=dpi,
                    quality=quality,
                    overwrite=overwrite,
                )
                summary.converted_pages += converted
                summary.skipped_pages += skipped
            except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError, OSError, ValueError) as exc:
                summary.failed_pdfs += 1
                summary.failures.append(f"{pdf_path.name}: {exc}")
                progress.console.print(f"[red]Failed[/red] {pdf_path.name}: {exc}")
            progress.advance(overall_task)

    _render_summary(console, summary)
    return summary


def _render_summary(console: Console, summary: ConversionSummary) -> None:
    """Render a compact conversion summary."""

    table = Table(title="Conversion Summary", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Charts directory", str(summary.charts_dir))
    table.add_row("Output directory", str(summary.output_dir))
    table.add_row("PDFs processed", str(summary.total_pdfs))
    table.add_row("Pages converted", str(summary.converted_pages))
    table.add_row("Pages skipped", str(summary.skipped_pages))
    table.add_row("Failed PDFs", str(summary.failed_pdfs))
    console.print(table)

    if summary.failures:
        failure_table = Table(title="Failures", expand=False)
        failure_table.add_column("PDF", style="bold red")
        failure_table.add_column("Error")
        for failure in summary.failures:
            pdf_name, _, message = failure.partition(": ")
            failure_table.add_row(pdf_name, message)
        console.print(failure_table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ADIP chart PDFs into JPEG images."
    )
    parser.add_argument(
        "charts_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_CHARTS_DIR,
        help=f"Directory containing chart PDFs (default: {DEFAULT_CHARTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where JPEGs will be saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render resolution in DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality from 1 to 95 (default: {DEFAULT_JPEG_QUALITY})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPEGs instead of skipping them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        convert_charts(
            args.charts_dir,
            args.output_dir,
            dpi=args.dpi,
            quality=args.quality,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
