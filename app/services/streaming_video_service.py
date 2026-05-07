from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from app import pipeline
from app.config import Settings, get_settings
from app.models import Storyline, SummaryReport, VideoMetadata


@dataclass
class StreamVideoResult:
    source: str
    status: str
    video_id: str | None = None
    metadata: VideoMetadata | None = None
    summary: SummaryReport | None = None
    storyline: Storyline | None = None
    error: str | None = None


@dataclass
class StreamBatchState:
    total: int
    events: list[dict] = field(default_factory=list)
    results: list[StreamVideoResult] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for result in self.results if result.status == "succeeded")

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if result.status == "failed")


class StreamingVideoService:
    """Two-stage URL pipeline: ingest next video while analyzing prior videos."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def process_urls(self, urls: list[str], query: str | None = None) -> Iterator[StreamBatchState]:
        clean_urls = [url.strip() for url in urls if url and url.strip()]
        state = StreamBatchState(total=len(clean_urls))
        if not clean_urls:
            state.events.append({"stage": "error", "message": "没有可处理的 URL。"})
            yield state
            return

        event_queue: queue.Queue[dict] = queue.Queue()
        analysis_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        ingest_settings = self.settings.model_copy(deep=True)
        analysis_settings = self.settings.model_copy(deep=True)

        def emit(event: dict) -> None:
            event_queue.put(event)

        def ingest_worker() -> None:
            for index, url in enumerate(clean_urls, start=1):
                emit({"stage": "ingest_started", "source": url, "index": index, "message": f"阶段一开始：{url}"})
                try:
                    video_id = pipeline.ingest_url(url, ingest_settings)
                    emit({"stage": "ingest_done", "source": url, "video_id": video_id, "index": index, "message": f"阶段一完成：{video_id}"})
                    analysis_queue.put((url, video_id))
                except Exception as exc:  # noqa: BLE001
                    emit({"stage": "failed", "source": url, "index": index, "error": str(exc), "message": f"阶段一失败：{exc}"})
            analysis_queue.put(None)

        def analysis_worker() -> None:
            while True:
                item = analysis_queue.get()
                if item is None:
                    return
                source, video_id = item
                emit({"stage": "analysis_started", "source": source, "video_id": video_id, "message": f"后续分析开始：{video_id}"})
                try:
                    result = pipeline.process_existing(video_id, query=query, export_to_obsidian=False, settings=analysis_settings)
                    metadata = pipeline.read_metadata(video_id, analysis_settings)
                    emit(
                        {
                            "stage": "analysis_done",
                            "source": source,
                            "video_id": video_id,
                            "result": StreamVideoResult(
                                source=source,
                                status="succeeded",
                                video_id=video_id,
                                metadata=metadata,
                                summary=result.get("summary") if isinstance(result.get("summary"), SummaryReport) else None,
                                storyline=result.get("storyline") if isinstance(result.get("storyline"), Storyline) else None,
                            ),
                            "message": f"后续分析完成：{video_id}",
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    emit(
                        {
                            "stage": "failed",
                            "source": source,
                            "video_id": video_id,
                            "result": StreamVideoResult(source=source, status="failed", video_id=video_id, error=str(exc)),
                            "error": str(exc),
                            "message": f"后续分析失败：{video_id} {exc}",
                        }
                    )

        ingest_thread = threading.Thread(target=ingest_worker, name="vka-stream-ingest", daemon=True)
        analysis_thread = threading.Thread(target=analysis_worker, name="vka-stream-analysis", daemon=True)
        ingest_thread.start()
        analysis_thread.start()

        while ingest_thread.is_alive() or analysis_thread.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            state.events.append(_event_for_display(event))
            result = event.get("result")
            if isinstance(result, StreamVideoResult):
                state.results.append(result)
            yield state

        yield state


def parse_url_lines(value: str | None) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def local_file_path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _event_for_display(event: dict) -> dict:
    event = dict(event)
    result = event.pop("result", None)
    if isinstance(result, StreamVideoResult):
        event["result"] = {
            "source": result.source,
            "status": result.status,
            "video_id": result.video_id,
            "error": result.error,
        }
    return event

