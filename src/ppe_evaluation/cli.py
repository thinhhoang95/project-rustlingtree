from __future__ import annotations

import argparse
import json
from pathlib import Path

from ppe_evaluation.gui import run_gui
from ppe_evaluation.metrics import evaluate_run


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate_run(
        args.run_dir,
        args.ground_truth,
        args.output_dir,
        class_aware=not args.class_agnostic,
    )
    print(json.dumps(report.to_json_dict(), indent=2))
    return 0


def _gui(args: argparse.Namespace) -> int:
    print(f"Starting PPE grader at http://{args.host}:{args.port}")
    run_gui(args.run_dir, args.ground_truth, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppe-eval", description="Evaluate and grade PPE run artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one completed PPE run against ground truth")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--ground-truth", type=Path, default=None)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument(
        "--class-agnostic",
        action="store_true",
        help="Match windows by medoid and IoU only, ignoring class_name.",
    )
    evaluate.set_defaults(func=_evaluate)

    gui = subparsers.add_parser("gui", help="Launch the browser grader for one PPE run")
    gui.add_argument("--run-dir", required=True, type=Path)
    gui.add_argument("--ground-truth", type=Path, default=None)
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.set_defaults(func=_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
