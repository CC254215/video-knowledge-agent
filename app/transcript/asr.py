from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable

import httpx
import logging

from app.config import Settings, get_settings
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_LOCAL_ASR_LOCK = threading.Lock()
_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORIES: set[str] = set()


class ASRUnavailableError(RuntimeError):
    pass


class ASRAdapter:
    def transcribe(
        self,
        media_path: str | Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[dict[str, object]]:
        raise NotImplementedError


class FasterWhisperASRAdapter(ASRAdapter):
    def __init__(
        self,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe(
        self,
        media_path: str | Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[dict[str, object]]:
        model: Any
        try:
            model = _get_whisper_model(self.model_name, self.device, self.compute_type)
        except RuntimeError as exc:
            if not _is_cuda_runtime_error(exc):
                raise
            logger.warning("CUDA ASR initialization failed; falling back to CPU int8: %s", exc)
            model = _get_whisper_model(self.model_name, "cpu", "int8")

        try:
            with _LOCAL_ASR_LOCK:
                segments, info = model.transcribe(
                    str(media_path),
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                )
                items = _consume_segments(segments, info, progress_callback)
        except RuntimeError as exc:
            if not _is_cuda_runtime_error(exc) or self.device == "cpu":
                raise
            logger.warning("CUDA ASR inference failed; retrying with CPU int8: %s", exc)
            cpu_model = _get_whisper_model(self.model_name, "cpu", "int8")
            with _LOCAL_ASR_LOCK:
                segments, info = cpu_model.transcribe(
                    str(media_path),
                    beam_size=self.beam_size,
                    vad_filter=self.vad_filter,
                )
                items = _consume_segments(segments, info, progress_callback)

        if progress_callback:
            progress_callback("正在进行语音识别 100% · 转写完成")
        return items


def _get_whisper_model(model_name: str, device: str, compute_type: str) -> Any:
    if device in {"cuda", "auto"}:
        _configure_cuda_dll_search_path()
    key = (model_name, device, compute_type)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError("faster-whisper is not installed. Install with `pip install .[asr]`.") from exc
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _MODEL_CACHE[key] = model
        return model


def _configure_cuda_dll_search_path() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import site
    except ImportError:
        return
    candidates: list[Path] = []
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        base = Path(root) / "nvidia"
        candidates.extend((base / "cublas" / "bin", base / "cudnn" / "bin", base / "cuda_nvrtc" / "bin"))
    registered = {str(path).lower() for path in candidates if path.exists()}
    current_path = os.environ.get("PATH", "")
    current_parts = {part.lower() for part in current_path.split(os.pathsep) if part}
    missing_from_path = [path for path in sorted(registered) if path not in current_parts]
    if missing_from_path:
        os.environ["PATH"] = os.pathsep.join([*missing_from_path, current_path])
    for path_text in sorted(registered):
        if path_text in _DLL_DIRECTORIES:
            continue
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path_text))
        _DLL_DIRECTORIES.add(path_text)


def _consume_segments(
    segments: Any,
    info: Any,
    progress_callback: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    items: list[dict[str, object]] = []
    last_percent = -1
    for segment in segments:
        items.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text, "asr_confidence": None})
        if progress_callback and duration:
            percent = min(100, int(float(segment.end) * 100 / duration))
            if percent != last_percent:
                last_percent = percent
                progress_callback(
                    f"正在进行语音识别 {percent:.1f}% · 已转写 {float(segment.end) / 60:.1f}/{duration / 60:.1f} 分钟"
                )
    return items


def _is_cuda_runtime_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("cuda", "cublas", "cudnn", "out of memory"))


class OpenAICompatibleASRAdapter(ASRAdapter):
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def transcribe(
        self,
        media_path: str | Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[dict[str, object]]:
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

        if progress_callback:
            progress_callback("正在上传音频并等待云端语音识别")
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
        compute_type = "int8" if settings.asr_device == "cpu" and settings.asr_compute_type == "auto" else settings.asr_compute_type
        return FasterWhisperASRAdapter(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=compute_type,
            beam_size=settings.asr_beam_size,
            vad_filter=settings.asr_vad_filter,
        )
    if not settings.has_cloud_asr_config:
        logger.warning(
            "ASR_PROVIDER=%s but cloud ASR config is incomplete; falling back to local faster-whisper.",
            settings.asr_provider,
        )
        compute_type = "int8" if settings.asr_device == "cpu" and settings.asr_compute_type == "auto" else settings.asr_compute_type
        return FasterWhisperASRAdapter(
            settings.asr_model,
            device=settings.asr_device,
            compute_type=compute_type,
            beam_size=settings.asr_beam_size,
            vad_filter=settings.asr_vad_filter,
        )
    return OpenAICompatibleASRAdapter(settings)


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"
