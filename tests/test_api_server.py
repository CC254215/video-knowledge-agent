from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.server import create_app
from app.config import Settings
from app.models import ConfidenceLevel, ConversationTurn, SummaryReport, VideoMetadata
from app.services.video_service import ProcessVideoResult


def test_process_video_task_succeeds(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")

    def fake_process_video(self, url, uploaded_file_path, query=None, progress_callback=None):
        if progress_callback:
            progress_callback("fake progress")
        return ProcessVideoResult(
            success=True,
            video_id="v1",
            metadata=VideoMetadata(video_id="v1", title="Video One"),
            summary=_summary("v1"),
            storyline=None,
            suggested_questions=["q1"],
            obsidian_status="skipped",
        )

    monkeypatch.setattr("app.api.server.VideoService.process_video", fake_process_video)
    client = TestClient(create_app(settings))

    created = client.post("/api/videos/process", json={"url": "https://example.com/video"}).json()
    task = _wait_for_task(client, created["task_id"])

    assert task["status"] == "succeeded"
    assert task["result"]["video_id"] == "v1"
    assert task["progress"][0]["message"] == "fake progress"


def test_process_video_task_records_failure(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")

    def fake_process_video(self, url, uploaded_file_path, query=None, progress_callback=None):
        return ProcessVideoResult(success=False, error="real pipeline error")

    monkeypatch.setattr("app.api.server.VideoService.process_video", fake_process_video)
    client = TestClient(create_app(settings))

    created = client.post("/api/videos/process", json={"url": "https://example.com/video"}).json()
    task = _wait_for_task(client, created["task_id"])

    assert task["status"] == "failed"
    assert task["error"] == "real pipeline error"


def test_process_video_task_records_heartbeat(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")
    monkeypatch.setattr("app.api.server.API_PROGRESS_HEARTBEAT_SECONDS", 0.01)

    def fake_process_video(self, url, uploaded_file_path, query=None, progress_callback=None):
        time.sleep(0.05)
        return ProcessVideoResult(success=False, error="slow failure")

    monkeypatch.setattr("app.api.server.VideoService.process_video", fake_process_video)
    client = TestClient(create_app(settings))

    created = client.post("/api/videos/process", json={"url": "https://example.com/video"}).json()
    task = _wait_for_task(client, created["task_id"])

    assert task["status"] == "failed"
    assert any("任务仍在运行" in item["message"] for item in task["progress"])


def test_video_read_endpoints_and_ask(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")
    video_dir = tmp_path / "videos" / "v1"
    video_dir.mkdir(parents=True)
    metadata = VideoMetadata(video_id="v1", title="Video One")
    (video_dir / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    (video_dir / "summary.json").write_text(_summary("v1").model_dump_json(), encoding="utf-8")
    (video_dir / "multimodal_segments.json").write_text(
        json.dumps([{"segment_id": "seg_0", "start": 0, "end": 10, "transcript_text": "evidence"}]),
        encoding="utf-8",
    )

    def fake_ask(self, video_id, question, conversation_id=None):
        return type(
            "Answer",
            (),
            {
                "answer": "grounded",
                "evidence": [{"segment_id": "seg_0"}],
                "timestamps": ["00:00:00"],
                "evidence_types": ["speech"],
                "confidence": "medium",
                "needs_visual_check": False,
                "reason": "test",
                "suggested_followup_questions": [],
                "raw": ConversationTurn(
                    turn_id="t1",
                    video_id=video_id,
                    user_question=question,
                    answer="grounded",
                    evidence=[{"segment_id": "seg_0"}],
                    timestamps=["00:00:00"],
                    evidence_types=["speech"],
                    confidence=ConfidenceLevel.medium,
                    needs_visual_check=False,
                    reason="test",
                ),
                "error": None,
            },
        )()

    monkeypatch.setattr("app.api.server.ChatService.ask", fake_ask)
    client = TestClient(create_app(settings))

    assert client.get("/api/videos/v1").json()["metadata"]["video_id"] == "v1"
    assert client.get("/api/videos/v1/summary").json()["video_id"] == "v1"
    assert client.get("/api/videos/v1/evidence").json()["segments"][0]["segment_id"] == "seg_0"
    assert client.post("/api/videos/v1/ask", json={"question": "what?"}).json()["answer"] == "grounded"


def _summary(video_id: str) -> SummaryReport:
    return SummaryReport(
        video_id=video_id,
        quick_overview=["overview"],
        structured_outline=[],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=[],
        modality_note="test",
        confidence=ConfidenceLevel.medium,
    )


def _wait_for_task(client: TestClient, task_id: str) -> dict:
    for _ in range(50):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"succeeded", "failed"}:
            return task
        time.sleep(0.02)
    raise AssertionError("task did not finish")
