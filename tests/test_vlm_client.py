from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vlm_ppe.agents.vlm_client import OpenRouterVLMClient
from vlm_ppe.schemas import EvidenceImage, KMetric


def test_openrouter_client_sends_multimodal_json_request(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "panel.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    calls: dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            calls["request"] = kwargs
            content = json.dumps(
                {
                    "chosen_k": 2,
                    "confidence": 0.9,
                    "rationale": ["K=2 is visually distinct."],
                    "rejected_alternatives": [],
                    "clusters_to_recheck": [],
                    "retry_requested": False,
                    "requested_k_max": None,
                    "suggested_action": "accept",
                }
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            calls["client"] = {"api_key": api_key, "base_url": base_url}
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = OpenRouterVLMClient(
        model="google/gemini-2.5-flash",
        app_referer="https://example.test",
        app_title="VLM PPE Test",
    )
    review = client.review_clusters(
        evidence_images=[
            EvidenceImage(kind="cluster_panel", path=image_path.as_posix(), caption="Cluster panel")
        ],
        metrics=[KMetric(k=2, inertia=1.0, silhouette=0.5, cluster_count_min=1, cluster_count_max=3, cluster_count_mean=2.0)],
        available_k=[1, 2],
        attempt=0,
        max_retries=1,
        prompt="Return JSON.",
    )

    assert review.chosen_k == 2
    assert calls["client"] == {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"}
    request = calls["request"]
    assert request["model"] == "google/gemini-2.5-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_headers"] == {
        "HTTP-Referer": "https://example.test",
        "X-OpenRouter-Title": "VLM PPE Test",
    }
    content = request["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Return JSON."}
    assert content[1] == {"type": "text", "text": "Image: Cluster panel"}
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openrouter_client_accepts_single_string_rationale(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "panel.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            content = json.dumps(
                {
                    "chosen_k": 4,
                    "confidence": 0.9,
                    "rationale": "K=4 is visually distinct.",
                    "rejected_alternatives": "K=5 splits a coherent group.",
                    "clusters_to_recheck": [],
                    "retry_requested": False,
                    "requested_k_max": None,
                    "suggested_action": "accept",
                }
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = OpenRouterVLMClient(model="google/gemini-2.5-flash")
    review = client.review_clusters(
        evidence_images=[EvidenceImage(kind="cluster_panel", path=image_path.as_posix(), caption="Cluster panel")],
        metrics=[],
        available_k=[4],
        attempt=0,
        max_retries=1,
        prompt="Return JSON.",
    )

    assert review.rationale == ["K=4 is visually distinct."]
    assert review.rejected_alternatives == ["K=5 splits a coherent group."]


def test_openrouter_client_reviews_windows(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "windows.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: dict[str, Any] = {}

    class FakeCompletions:
        def create(self, **kwargs: Any) -> Any:
            calls["request"] = kwargs
            content = json.dumps(
                {
                    "cluster_id": 2,
                    "windows": [
                        {
                            "window_id": "C2_W1",
                            "start_station_index": 3,
                            "end_station_index": 6,
                            "class_name": "dogleg",
                            "confidence": 0.82,
                            "visual_reason": "One outward excursion is visible.",
                        }
                    ],
                    "outlier_notes": [],
                    "suggested_action": "accept",
                }
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = OpenRouterVLMClient(model="google/gemini-2.5-flash")
    review = client.review_windows(
        cluster_id=2,
        evidence_images=[EvidenceImage(kind="residual_windows", path=image_path.as_posix(), caption="Cluster 2 windows")],
        prompt="Classify windows.",
    )

    assert review.cluster_id == 2
    assert review.windows[0].class_name == "dogleg"
    assert calls["request"]["messages"][0]["content"][0] == {"type": "text", "text": "Classify windows."}
