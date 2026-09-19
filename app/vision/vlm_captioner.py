from __future__ import annotations

import base64
import json
import logging
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from app.config import Settings, get_settings
from app.models import FrameCaption, VideoFrame
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)


class VLMCaptioner:
    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.runtime_vlm_model or "no-vision-model"

    def caption_frames(self, frames: list[VideoFrame]) -> list[FrameCaption]:
        if not self.settings.has_vision_config:
            logger.info("No vision model config; marking frame captions as missing_model.")
            return [self._empty_caption(frame, "no-vision-model", status="missing_model") for frame in frames]

        captions: list[FrameCaption | None] = [None] * len(frames)
        concurrency = max(1, min(len(frames) or 1, self.settings.vlm_caption_concurrency))
        logger.info("Captioning %s frame(s) with VLM concurrency=%s", len(frames), concurrency)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self._caption_one_with_retry, frame): index
                for index, frame in enumerate(frames)
            }
            for future in as_completed(futures):
                index = futures[future]
                frame = frames[index]
                try:
                    captions[index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Vision caption failed for %s after retries: %s", frame.frame_id, exc)
                    captions[index] = self._empty_caption(frame, self.model)
        return [caption for caption in captions if caption is not None]

    def caption_frame(self, image_path: str, context: str | None = None) -> FrameCaption:
        path = Path(image_path)
        frame = VideoFrame(frame_id=path.stem, timestamp=_timestamp_from_frame_name(path.stem), path=str(path), selected=True)
        if not self.settings.has_vision_config:
            return self._empty_caption(frame, "no-vision-model", status="missing_model")
        return self._caption_one(frame, context=context)

    def _caption_one_with_retry(self, frame: VideoFrame, context: str | None = None) -> FrameCaption:
        # Bounded concurrency can expose transient provider-side 429/5xx/timeout
        # errors. Retry with small exponential backoff so one flaky frame does
        # not fail or slow the entire video pipeline.
        attempts = max(1, self.settings.vlm_caption_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._caption_one(frame, context=context)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts - 1:
                    break
                time.sleep(min(4.0, 0.6 * (2**attempt)))
        raise last_error or RuntimeError("VLM caption failed")

    def _caption_one(self, frame: VideoFrame, context: str | None = None) -> FrameCaption:
        if not Path(frame.path).exists():
            return self._empty_caption(frame, self.model, status="failed")
        image_path = Path(frame.path)
        image_url, mime_type, image_size = _image_data_url(image_path)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请用简体中文描述这张视频帧，用于有证据约束的视频问答。"
                                "描述要简洁；如果只是人物出镜，简要说明即可。"
                                "如果画面包含幻灯片、代码、图表、公式、软件界面或演示步骤，重点描述可见的语义内容。"
                                "当前禁用 OCR，不要逐字转写微小文字，也不要推断画面中不可见的事实。"
                                f"附近语音上下文：{context or ''}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
        }
        def _request() -> httpx.Response:
            result = httpx.post(
                _join_openai_endpoint(self.settings.runtime_vlm_base_url, "chat/completions"),
                headers={"Authorization": f"Bearer {self.settings.vision_api_key}"},
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
            if result.status_code == 429 or result.status_code >= 500:
                raise RuntimeError(f"retryable_http_status={result.status_code}, body={result.text[:1000]}")
            return result

        response = get_scheduler(self.settings).call("vlm", _request)
        if response.status_code >= 400:
            body = response.text[:4000]
            safe_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text_length": len(payload["messages"][0]["content"][0]["text"])},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url_prefix": image_url[:64],
                                    "url_length": len(image_url),
                                },
                            },
                        ],
                    }
                ],
                "temperature": payload["temperature"],
            }
            logger.warning(
                "VLM caption failed: status=%s model=%s mime=%s image_size=%s body=%s payload=%s",
                response.status_code,
                self.model,
                mime_type,
                image_size,
                body,
                safe_payload,
            )
            raise RuntimeError(
                f"VLM request failed: status={response.status_code}, model={self.model}, "
                f"mime={mime_type}, image_size={image_size}, body={body}"
            )
        data = response.json()
        caption = data["choices"][0]["message"]["content"].strip()
        return FrameCaption(frame_id=frame.frame_id, timestamp=frame.timestamp, image_path=frame.path, caption=caption, model=self.model, confidence=None, status="success")

    def _empty_caption(self, frame: VideoFrame, model: str, status: str = "failed") -> FrameCaption:
        return FrameCaption(frame_id=frame.frame_id, timestamp=frame.timestamp, image_path=frame.path, caption="", model=model, confidence=None, status=status)


def _image_data_url(path: Path) -> tuple[str, str, int]:
    mime = mimetypes.guess_type(path.name)[0] or ("image/png" if path.suffix.lower() == ".png" else "image/jpeg")
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", mime, len(raw)


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"


def _timestamp_from_frame_name(name: str) -> float:
    digits = "".join(ch for ch in name if ch.isdigit())
    return float(int(digits)) if digits else 0.0


def save_frame_captions(captions: list[FrameCaption], video_dir: str | Path) -> Path:
    path = Path(video_dir) / "frame_captions.json"
    path.write_text(json.dumps([caption.model_dump() for caption in captions], ensure_ascii=False, indent=2), encoding="utf-8")
    return path
