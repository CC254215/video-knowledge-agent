from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import logging

from app.config import Settings, get_settings
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)


class ASRUnavailableError(RuntimeError):
    pass


class ASRAdapter:
    def transcribe(self, media_path: str | Path) -> list[dict[str, object]]:
        raise NotImplementedError


class FasterWhisperASRAdapter(ASRAdapter):
    def __init__(self, model_name: str = "small") -> None:
        self.model_name = model_name

    def transcribe(self, media_path: str | Path) -> list[dict[str, object]]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError("faster-whisper is not installed. Install with `pip install .[asr]`.") from exc

        model: Any
        try:
            # Try auto/default backend first; many machines with CUDA will use GPU.
            model = WhisperModel(self.model_name)
        except RuntimeError as exc:
            # If CUDA libs are missing, fall back to a CPU-safe configuration.
            if "cublas" not in str(exc).lower() and "cuda" not in str(exc).lower():
                raise
            model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

        try:
            segments, _info = model.transcribe(str(media_path))
        except RuntimeError as exc:
            # Some environments fail only at inference time; retry on CPU once.
            if "cublas" not in str(exc).lower() and "cuda" not in str(exc).lower():
                raise
            cpu_model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            segments, _info = cpu_model.transcribe(str(media_path))

        return [
            {"start": float(segment.start), "end": float(segment.end), "text": segment.text, "asr_confidence": None}
            for segment in segments
        ]


class OpenAICompatibleASRAdapter(ASRAdapter):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def transcribe(self, media_path: str | Path) -> list[dict[str, object]]:
        if not self.settings.has_cloud_asr_config:
            raise ASRUnavailableError("Cloud ASR config is incomplete. Set ASR_PROVIDER, ASR_API_KEY, ASR_BASE_URL, ASR_MODEL.")
        url = _join_openai_endpoint(self.settings.asr_base_url or "", "audio/transcriptions")
        def _request() -> httpx.Response:
            with Path(media_path).open("rb") as fh:
                return httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {self.settings.asr_api_key}"},
                    data={"model": self.settings.asr_model, "response_format": "verbose_json", "timestamp_granularities[]": "segment"},
                    files={"file": (Path(media_path).name, fh, "application/octet-stream")},
                    timeout=max(120.0, self.settings.request_timeout_seconds),
                )

        def _retryable_request() -> httpx.Response:
            result = _request()
            if result.status_code == 429 or result.status_code >= 500:
                raise RuntimeError(f"retryable_http_status={result.status_code}, body={result.text[:1000]}")
            return result

        response = get_scheduler(self.settings).call("asr", _retryable_request)
        response.raise_for_status()
        data = response.json()
        if data.get("segments"):
            return [
                {
                    "start": float(item.get("start", 0.0)),
                    "end": float(item.get("end", item.get("start", 0.0))),
                    "text": item.get("text", ""),
                    "asr_confidence": None,
                }
                for item in data["segments"]
                if item.get("text")
            ]
        text = data.get("text", "")
        return [{"start": 0.0, "end": 0.0, "text": text, "asr_confidence": None}] if text else []


def get_asr_adapter(settings: Settings | None = None) -> ASRAdapter:
    settings = settings or get_settings()
    provider = settings.asr_provider.lower()
    if provider in {"local", "faster-whisper", "faster_whisper", "whisper"}:
        return FasterWhisperASRAdapter(settings.asr_model)
    if not settings.has_cloud_asr_config:
        logger.warning(
            "ASR_PROVIDER=%s but cloud ASR config is incomplete; falling back to local faster-whisper.",
            settings.asr_provider,
        )
        return FasterWhisperASRAdapter(settings.asr_model)
    return OpenAICompatibleASRAdapter(settings)


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"
