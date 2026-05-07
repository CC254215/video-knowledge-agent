from __future__ import annotations

import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from app import pipeline
from app.api.task_store import APITaskStore
from app.config import Settings, get_settings
from app.models import MultimodalSegment, Storyline, SummaryReport
from app.services.chat_service import ChatService
from app.services.video_service import VideoService
from app.transcript.multimodal_segmenter import load_multimodal_segments

API_PROGRESS_HEARTBEAT_SECONDS = 15.0


class ProcessVideoRequest(BaseModel):
    url: str | None = None
    file_path: str | None = None
    query: str | None = None

    @model_validator(mode="after")
    def require_source(self) -> "ProcessVideoRequest":
        if not (self.url or "").strip() and not (self.file_path or "").strip():
            raise ValueError("Provide either url or file_path.")
        return self


class ProcessTaskResponse(BaseModel):
    task_id: str
    status: str
    task_url: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None


class APIRuntime:
    def __init__(self, settings: Settings | None = None, max_workers: int = 2) -> None:
        self.settings = settings or get_settings()
        self.task_store = APITaskStore(self.settings.data_dir / "vka_api_tasks.sqlite3")
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vka-api-worker")

    def submit_process(self, request: ProcessVideoRequest) -> dict[str, Any]:
        payload = request.model_dump()
        task_id = f"task_{uuid.uuid4().hex}"
        task = self.task_store.create_task(task_id, "process_video", payload)
        self.executor.submit(self._run_process_task, task_id, payload)
        return task

    def _run_process_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self.task_store.mark_running(task_id)
        heartbeat_done = threading.Event()
        started_at = time.monotonic()

        def progress(message: str) -> None:
            self.task_store.add_progress(task_id, message)

        def heartbeat() -> None:
            while not heartbeat_done.wait(API_PROGRESS_HEARTBEAT_SECONDS):
                elapsed = int(time.monotonic() - started_at)
                self.task_store.add_progress(task_id, f"任务仍在运行，已用时 {elapsed}s；当前阶段可能正在等待模型或外部服务返回")

        heartbeat_thread = threading.Thread(target=heartbeat, name=f"vka-api-heartbeat-{task_id}", daemon=True)
        heartbeat_thread.start()

        try:
            result = VideoService(self.settings).process_video(
                url=payload.get("url"),
                uploaded_file_path=payload.get("file_path"),
                query=payload.get("query"),
                progress_callback=progress,
            )
            if not result.success:
                self.task_store.mark_failed(task_id, result.error or "video_processing_failed")
                return
            self.task_store.mark_succeeded(task_id, _process_result_payload(result))
        except Exception as exc:  # noqa: BLE001
            self.task_store.mark_failed(task_id, str(exc).splitlines()[0][:4000])
        finally:
            heartbeat_done.set()


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime = APIRuntime(settings)
    app = FastAPI(title="Video Knowledge Agent API", version="0.1.0")
    app.state.runtime = runtime

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        config = runtime.settings.runtime_config_summary()
        return {
            "status": "ok",
            "has_llm_config": runtime.settings.has_llm_config,
            "has_embedding_config": runtime.settings.has_embedding_config,
            "has_vision_config": runtime.settings.has_vision_config,
            "retrieval_mode": config.get("RETRIEVAL_MODE"),
            "embedding_cache_enabled": config.get("EMBEDDING_CACHE_ENABLED"),
        }

    @app.post("/api/videos/process", response_model=ProcessTaskResponse)
    def process_video(request: ProcessVideoRequest) -> ProcessTaskResponse:
        task = runtime.submit_process(request)
        return ProcessTaskResponse(
            task_id=task["task_id"],
            status=task["status"],
            task_url=f"/api/tasks/{task['task_id']}",
        )

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = runtime.task_store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task_not_found")
        return task

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, Any]:
        try:
            metadata = pipeline.read_metadata(video_id, runtime.settings)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"video_not_found: {exc}") from exc
        video_path = pipeline.video_dir(video_id, runtime.settings)
        return {
            "metadata": jsonable_encoder(metadata),
            "has_summary": (video_path / "summary.json").exists(),
            "has_storyline": (video_path / "storyline.json").exists(),
            "has_multimodal_segments": (video_path / "multimodal_segments.json").exists(),
            "video_dir": str(video_path),
        }

    @app.get("/api/videos/{video_id}/summary")
    def get_summary(video_id: str) -> dict[str, Any]:
        return jsonable_encoder(_read_summary(video_id, runtime.settings))

    @app.get("/api/videos/{video_id}/storyline")
    def get_storyline(video_id: str) -> dict[str, Any]:
        return jsonable_encoder(_read_storyline(video_id, runtime.settings))

    @app.get("/api/videos/{video_id}/evidence")
    def get_evidence(video_id: str) -> dict[str, Any]:
        segments = _read_multimodal_segments(video_id, runtime.settings)
        return {"video_id": video_id, "segments": jsonable_encoder(segments)}

    @app.post("/api/videos/{video_id}/ask")
    def ask_video(video_id: str, request: AskRequest) -> dict[str, Any]:
        result = ChatService(runtime.settings).ask(video_id, request.question, conversation_id=request.conversation_id)
        if result.error:
            raise HTTPException(status_code=400, detail=jsonable_encoder(result))
        return jsonable_encoder(_chat_result_payload(result))

    @app.get("/api/videos/{video_id}/frames/{frame_id}")
    def get_frame(video_id: str, frame_id: str) -> FileResponse:
        path = _find_frame_path(video_id, frame_id, runtime.settings)
        if not path:
            raise HTTPException(status_code=404, detail="frame_not_found")
        return FileResponse(path)

    return app


def _process_result_payload(result: Any) -> dict[str, Any]:
    return {
        "success": result.success,
        "video_id": result.video_id,
        "metadata": jsonable_encoder(result.metadata),
        "summary": jsonable_encoder(result.summary),
        "storyline": jsonable_encoder(result.storyline),
        "modality_profile": jsonable_encoder(result.modality_profile),
        "suggested_questions": result.suggested_questions or [],
        "obsidian_path": result.obsidian_path,
        "obsidian_status": result.obsidian_status,
    }


def _chat_result_payload(result: Any) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "evidence": result.evidence,
        "timestamps": result.timestamps,
        "evidence_types": result.evidence_types,
        "confidence": result.confidence,
        "needs_visual_check": result.needs_visual_check,
        "reason": result.reason,
        "suggested_followup_questions": result.suggested_followup_questions,
        "raw": result.raw,
    }


def _read_summary(video_id: str, settings: Settings) -> SummaryReport:
    path = pipeline.video_dir(video_id, settings) / "summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="summary_not_found")
    return SummaryReport.model_validate_json(path.read_text(encoding="utf-8"))


def _read_storyline(video_id: str, settings: Settings) -> Storyline:
    path = pipeline.video_dir(video_id, settings) / "storyline.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="storyline_not_found")
    return Storyline.model_validate_json(path.read_text(encoding="utf-8"))


def _read_multimodal_segments(video_id: str, settings: Settings) -> list[MultimodalSegment]:
    path = pipeline.video_dir(video_id, settings) / "multimodal_segments.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="multimodal_segments_not_found")
    return load_multimodal_segments(path)


def _find_frame_path(video_id: str, frame_id: str, settings: Settings) -> Path | None:
    video_path = pipeline.video_dir(video_id, settings)
    candidates = list((video_path / "frames").rglob(f"{frame_id}.*")) if (video_path / "frames").exists() else []
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
