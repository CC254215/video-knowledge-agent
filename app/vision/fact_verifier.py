from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models import VideoFrame
from app.runtime.scheduler import get_scheduler
from app.vision.vlm_captioner import _image_data_url, _join_openai_endpoint


class VisualFactVerification(BaseModel):
    fact: str
    verification_type: Literal["single_frame", "temporal_sequence", "stage"] = "single_frame"
    status: Literal["supported", "unsupported", "unclear"] = "unclear"
    observed_fact: str = ""
    evidence_frame_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    reason: str = ""


class FactVerifier:
    """Ask a VLM to verify a known missing fact from already selected frames."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.runtime_vlm_model or "no-vision-model"

    def verify(self, fact: str, frames: list[VideoFrame], question_context: str = "", verification_type: str = "single_frame") -> VisualFactVerification:
        selected = sorted(frames, key=lambda item: item.timestamp)[:3]
        kind = verification_type if verification_type in {"single_frame", "temporal_sequence", "stage"} else "single_frame"
        if not selected or not self.settings.has_vision_config:
            return VisualFactVerification(fact=fact, verification_type=kind, reason="没有可用候选帧或视觉模型。")
        content = [{"type": "text", "text": (
            "先逐帧描述直接可见内容，再判断它是否支持目标事实。禁止根据问题补全、猜测或使用画面外知识。"
            f"\n问题上下文：{question_context}\n目标事实：{fact}\n核验类型：{kind}\n"
            "输出 JSON：status(supported|unsupported|unclear), observed_fact, evidence_frame_ids, confidence(low|medium|high), reason。"
        )}]
        for frame in selected:
            url, _, _ = _image_data_url(Path(frame.path))
            content.append({"type": "text", "text": f"frame_id={frame.frame_id}, timestamp={frame.timestamp:.2f}s"})
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0.0}
        try:
            response = get_scheduler(self.settings).call("vlm", lambda: httpx.post(
                _join_openai_endpoint(self.settings.runtime_vlm_base_url, "chat/completions"),
                headers={"Authorization": f"Bearer {self.settings.vision_api_key}"},
                json=payload, timeout=self.settings.request_timeout_seconds,
            ))
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            if isinstance(raw, str):
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                data = json.loads(cleaned)
            else:
                data = raw
            valid_ids = {frame.frame_id for frame in selected}
            ids = [str(item) for item in data.get("evidence_frame_ids", []) if str(item) in valid_ids]
            status = str(data.get("status", "unclear")).lower()
            if status not in {"supported", "unsupported", "unclear"}:
                status = "unclear"
            confidence = str(data.get("confidence", "low")).lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            return VisualFactVerification(
                fact=fact, verification_type=kind, status=status, observed_fact=str(data.get("observed_fact", "")),
                evidence_frame_ids=ids, confidence=confidence, reason=str(data.get("reason", "")),
            )
        except Exception as exc:  # noqa: BLE001
            return VisualFactVerification(fact=fact, verification_type=kind, reason=f"视觉事实核验失败: {exc}")
