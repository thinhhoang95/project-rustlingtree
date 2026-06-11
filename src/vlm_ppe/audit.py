from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vlm_ppe.schemas import ClusterReview, EvidenceImage, KMetric

LOGGER_NAME = "vlm_ppe.audit"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return value


def setup_audit_logging(run_dir: str | Path, *, level: str = "INFO", console: bool = True) -> logging.Logger:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time_gmt

    file_handler = logging.FileHandler(root / "audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("[vlm-ppe] %(message)s"))
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)

    logger.info("audit logging initialized run_dir=%s level=%s console=%s", root.as_posix(), level.upper(), console)
    return logger


def time_gmt(*args: Any) -> Any:
    return datetime.now(UTC).timetuple()


def get_audit_logger(run_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and run_dir is not None:
        setup_audit_logging(run_dir)
    return logger


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, separators=(",", ":"), ensure_ascii=False)
        stream.write("\n")


def _payload_summary(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    parts: list[str] = []
    for key, value in payload.items():
        if key in {"clustering_metrics", "evidence_images", "medoids", "config"}:
            if isinstance(value, list):
                parts.append(f"{key}=<{len(value)} items>")
            else:
                parts.append(f"{key}=<object>")
            continue
        if isinstance(value, str) and len(value) > 120:
            parts.append(f"{key}={value[:117]}...")
        else:
            parts.append(f"{key}={value!r}")
    return " ".join(parts)


def log_graph_event(
    *,
    run_dir: str | Path,
    node: str,
    status: str,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    event = {
        "timestamp_utc": utc_now_iso(),
        "node": node,
        "status": status,
        "message": message,
        "payload": payload or {},
    }
    root = Path(run_dir)
    _append_jsonl(root / "graph_events.jsonl", event)

    logger = get_audit_logger(root)
    summary = _payload_summary(payload or {})
    text = f"{node}: {status}"
    if message:
        text = f"{text} - {message}"
    if summary:
        text = f"{text} | {summary}"
    if status == "failed":
        logger.error(text)
    elif status == "started":
        logger.info(text)
    else:
        logger.info(text)


def _image_record(image: EvidenceImage) -> dict[str, Any]:
    path = Path(image.path)
    return {
        "kind": image.kind,
        "path": image.path,
        "caption": image.caption,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def log_vlm_request(
    *,
    run_dir: str | Path,
    attempt: int,
    model: str,
    prompt: str,
    evidence_images: list[EvidenceImage],
    metrics: list[KMetric],
    available_k: list[int],
    offline_override: bool,
) -> None:
    root = Path(run_dir)
    review_dir = root / "vlm_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    image_records = [_image_record(image) for image in evidence_images]
    request_payload = {
        "timestamp_utc": utc_now_iso(),
        "event": "vlm_request",
        "attempt": attempt,
        "model": model,
        "offline_override": offline_override,
        "available_k": available_k,
        "prompt_path": (review_dir / f"attempt_{attempt:02d}_prompt.txt").as_posix(),
        "images": image_records,
        "metrics": [metric.model_dump() for metric in metrics],
    }
    (review_dir / f"attempt_{attempt:02d}_prompt.txt").write_text(prompt, encoding="utf-8")
    with (review_dir / f"attempt_{attempt:02d}_request.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe({**request_payload, "prompt": prompt}), stream, indent=2, ensure_ascii=False)
    _append_jsonl(root / "vlm_interactions.jsonl", request_payload)

    logger = get_audit_logger(root)
    mode = "offline override" if offline_override else "OpenRouter request"
    logger.info("VLM attempt %02d prepared: %s model=%s available_k=%s images=%d", attempt, mode, model, available_k, len(image_records))
    for index, image in enumerate(image_records, start=1):
        logger.info(
            "VLM image %02d/%02d kind=%s exists=%s bytes=%s path=%s caption=%s",
            index,
            len(image_records),
            image["kind"],
            image["exists"],
            image["bytes"],
            image["path"],
            image["caption"],
        )


def log_vlm_response(
    *,
    run_dir: str | Path,
    attempt: int,
    review: ClusterReview,
    response_path: str,
) -> None:
    root = Path(run_dir)
    payload = {
        "timestamp_utc": utc_now_iso(),
        "event": "vlm_response",
        "attempt": attempt,
        "response_path": response_path,
        "review": review.model_dump(),
    }
    _append_jsonl(root / "vlm_interactions.jsonl", payload)
    logger = get_audit_logger(root)
    logger.info(
        "VLM response attempt %02d: chosen_k=%s confidence=%.3f action=%s retry=%s response=%s",
        attempt,
        review.chosen_k,
        review.confidence,
        review.suggested_action,
        review.retry_requested,
        response_path,
    )
    for index, rationale in enumerate(review.rationale, start=1):
        logger.info("VLM rationale %02d: %s", index, rationale)
