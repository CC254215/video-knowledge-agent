from __future__ import annotations

import time
from pathlib import Path

from app.config import Settings
from app.models import ConfidenceLevel, SummaryReport, VideoMetadata
from app.services.streaming_video_service import StreamingVideoService, parse_url_lines


def test_parse_url_lines() -> None:
    assert parse_url_lines("https://a\n\n https://b ") == ["https://a", "https://b"]


def test_streaming_service_runs_ingest_then_post_ingest(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="noop")
    calls: list[str] = []

    def fake_ingest_url(url, settings):
        calls.append(f"ingest:{url}")
        return "v1" if url.endswith("1") else "v2"

    def fake_process_existing(video_id, query=None, export_to_obsidian=False, settings=None):
        calls.append(f"analysis:{video_id}")
        time.sleep(0.01)
        return {"video_id": video_id, "summary": _summary(video_id), "storyline": None}

    def fake_read_metadata(video_id, settings=None):
        return VideoMetadata(video_id=video_id, title=f"title {video_id}")

    monkeypatch.setattr("app.services.streaming_video_service.pipeline.ingest_url", fake_ingest_url)
    monkeypatch.setattr("app.services.streaming_video_service.pipeline.process_existing", fake_process_existing)
    monkeypatch.setattr("app.services.streaming_video_service.pipeline.read_metadata", fake_read_metadata)

    states = list(StreamingVideoService(settings).process_urls(["https://x/1", "https://x/2"], query="q"))
    final = states[-1]

    assert final.completed == 2
    assert final.failed == 0
    assert [result.video_id for result in final.results] == ["v1", "v2"]
    assert "ingest:https://x/2" in calls
    assert "analysis:v1" in calls


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
