from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_ppe.agents.graph import run_graph
from vlm_ppe.config import load_config


def _run_through_medoid(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.log_level is not None or args.quiet:
        updates = {}
        if args.log_level is not None:
            updates["log_level"] = args.log_level
        if args.quiet:
            updates["log_to_console"] = False
        config = config.model_copy(update=updates)
    result = run_graph(config, chosen_k=args.chosen_k, run_id=args.run_id)
    print(json.dumps({"status": result.get("status"), "run_dir": result.get("run_dir"), "state_path": result.get("state_path")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-ppe", description="VLM-led practical procedure extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-through-medoid", help="Run PPE stages 0-6 through medoid extraction")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--chosen-k", type=int, default=None, help="Offline/manual cluster-count override")
    run_parser.add_argument("--run-id", type=str, default=None)
    run_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    run_parser.add_argument("--quiet", action="store_true", help="Disable console audit logging; audit.log is still written")
    run_parser.set_defaults(func=_run_through_medoid)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
