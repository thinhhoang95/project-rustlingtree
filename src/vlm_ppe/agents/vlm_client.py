from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from vlm_ppe.agents.prompts import cluster_review_prompt, window_cluster_selection_prompt, window_review_prompt
from vlm_ppe.schemas import ClusterMedoid, ClusterReview, EvidenceImage, KMetric, WindowClusterSelection, WindowReview

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ClusterReviewClient(Protocol):
    def review_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        metrics: list[KMetric],
        available_k: list[int],
        attempt: int,
        max_retries: int,
        prompt: str | None = None,
    ) -> ClusterReview:
        ...

    def review_windows(
        self,
        *,
        cluster_id: int,
        evidence_images: list[EvidenceImage],
        attempt: int = 0,
        max_attempts: int = 3,
        previous_review_json: str | None = None,
        prompt: str | None = None,
    ) -> WindowReview:
        ...

    def review_window_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        medoids: list[ClusterMedoid],
        prompt: str | None = None,
    ) -> WindowClusterSelection:
        ...


class OpenRouterVLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = "medium",
        app_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for VLM-led runs")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL
        self.app_referer = app_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self.app_title = app_title or os.environ.get("OPENROUTER_X_TITLE") or "project-rustlingtree/vlm-ppe"

    def review_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        metrics: list[KMetric],
        available_k: list[int],
        attempt: int,
        max_retries: int,
        prompt: str | None = None,
    ) -> ClusterReview:
        resolved_prompt = prompt or cluster_review_prompt(metrics, available_k, attempt, max_retries)
        text = self._request_json_text(prompt=resolved_prompt, evidence_images=evidence_images)
        return ClusterReview.model_validate(json.loads(text))

    def review_windows(
        self,
        *,
        cluster_id: int,
        evidence_images: list[EvidenceImage],
        attempt: int = 0,
        max_attempts: int = 3,
        previous_review_json: str | None = None,
        prompt: str | None = None,
    ) -> WindowReview:
        resolved_prompt = prompt or window_review_prompt(
            cluster_id,
            attempt=attempt,
            max_attempts=max_attempts,
            previous_review_json=previous_review_json,
        )
        text = self._request_json_text(prompt=resolved_prompt, evidence_images=evidence_images)
        return WindowReview.model_validate(json.loads(text))

    def review_window_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        medoids: list[ClusterMedoid],
        prompt: str | None = None,
    ) -> WindowClusterSelection:
        resolved_prompt = prompt or window_cluster_selection_prompt(medoids)
        text = self._request_json_text(prompt=resolved_prompt, evidence_images=evidence_images)
        return WindowClusterSelection.model_validate(json.loads(text))

    def _request_json_text(self, *, prompt: str, evidence_images: list[EvidenceImage]) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for OpenRouter VLM calls") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        contents: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in evidence_images:
            contents.append({"type": "text", "text": f"Image: {image.caption}"})
            contents.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(image.path))}})

        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": contents}],
            "response_format": {"type": "json_object"},
            "extra_headers": _openrouter_headers(self.app_referer, self.app_title),
        }
        if self.reasoning_effort is not None:
            request["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
        response = client.chat.completions.create(**request)
        text = _message_text(response.choices[0].message.content)
        if not text:
            raise ValueError("OpenRouter response did not contain text")
        return text


def _image_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _openrouter_headers(app_referer: str | None, app_title: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if app_referer:
        headers["HTTP-Referer"] = app_referer
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
    return headers


def _message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError("OpenRouter response did not contain text content")
