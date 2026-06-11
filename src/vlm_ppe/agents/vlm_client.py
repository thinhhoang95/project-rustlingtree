from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from vlm_ppe.agents.prompts import cluster_review_prompt
from vlm_ppe.schemas import ClusterReview, EvidenceImage, KMetric


class ClusterReviewClient(Protocol):
    def review_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        metrics: list[KMetric],
        available_k: list[int],
        attempt: int,
        max_retries: int,
    ) -> ClusterReview:
        ...


class GeminiVLMClient:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required unless --chosen-k is supplied")

    def review_clusters(
        self,
        *,
        evidence_images: list[EvidenceImage],
        metrics: list[KMetric],
        available_k: list[int],
        attempt: int,
        max_retries: int,
    ) -> ClusterReview:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        prompt = cluster_review_prompt(metrics, available_k, attempt, max_retries)
        contents: list[object] = [prompt]
        for image in evidence_images:
            image_path = Path(image.path)
            contents.append(f"Image: {image.caption}")
            contents.append(types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"))

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini response did not contain text")
        return ClusterReview.model_validate(json.loads(text))
