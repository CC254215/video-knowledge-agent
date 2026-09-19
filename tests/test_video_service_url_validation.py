import json
from pathlib import Path

from app.config import Settings
from app.models import ModalityMode, ModalityProfile, MultimodalSegment, OutlineItem, Storyline, SummaryReport, VideoMetadata
from app.services.video_service import ProcessVideoResult, VideoService, _apply_outline_time_ranges


def test_process_video_requires_url_or_file(tmp_path: Path):
    result = VideoService(Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"))).process_video("", None)

    assert result.success is False
    assert "请先输入视频 URL 或上传" in result.error


def test_process_video_prefers_uploaded_file_when_both_present(monkeypatch, tmp_path: Path):
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"))
    service = VideoService(settings)
    upload = tmp_path / "sample.mp4"
    upload.write_bytes(b"video")
    called = {"file": None, "url": None}

    def fake_run_pipeline(url=None, file_path=None, query=None, progress_callback=None):
        called["url"] = url
        called["file"] = file_path
        return ProcessVideoResult(
            success=True,
            video_id="v1",
            metadata=VideoMetadata(video_id="v1", title="Video"),
            summary=SummaryReport(
                video_id="v1",
                quick_overview=[],
                structured_outline=[],
                deep_analysis=[],
                action_items=[],
                important_quotes=[],
                open_questions=[],
                modality_note="text",
            ),
            storyline=Storyline(video_id="v1", query="q", nodes=[]),
            modality_profile=ModalityProfile(
                mode=ModalityMode.text_dominant,
                speech_density=1.0,
                visual_required_score=0,
                ocr_required_score=0,
                reason="text",
            ),
        )

    monkeypatch.setattr(service, "_run_pipeline", fake_run_pipeline)
    result = service.process_video("https://example.test/video", str(upload))

    assert result.success is True
    assert called["url"] is None
    assert called["file"] is not None


def test_load_latest_completed_restores_disk_context(tmp_path: Path):
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"))
    video_dir = settings.videos_dir / "v1"
    video_dir.mkdir(parents=True)
    metadata = VideoMetadata(video_id="v1", title="Restored video", duration=120)
    summary = SummaryReport(
        video_id="v1",
        quick_overview=["overview"],
        structured_outline=[],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=["question"],
        modality_note="text",
    )
    storyline = Storyline(video_id="v1", query="q", nodes=[])
    (video_dir / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")
    (video_dir / "summary.json").write_text(summary.model_dump_json(), encoding="utf-8")
    (video_dir / "storyline.json").write_text(storyline.model_dump_json(), encoding="utf-8")
    (video_dir / "processing_state.json").write_text(
        json.dumps({"stages": {"pipeline": {"status": "succeeded", "total_duration_seconds": 12}}}),
        encoding="utf-8",
    )

    result = VideoService(settings).load_latest_completed()

    assert result is not None
    assert result.video_id == "v1"
    assert result.summary.quick_overview == ["overview"]
    assert result.processing_state["stages"]["pipeline"]["status"] == "succeeded"


def test_outline_timestamp_is_shown_as_evidence_range():
    summary = SummaryReport(
        video_id="v1",
        quick_overview=[],
        structured_outline=[OutlineItem(timestamp="[00:00:00]", topic="topic", key_points=[], segment_ids=["s1", "s2"])],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=[],
        modality_note="text",
    )
    segments = [
        MultimodalSegment(segment_id="s1", start=0, end=100, transcript_text="a"),
        MultimodalSegment(segment_id="s2", start=100, end=220, transcript_text="b"),
    ]
    _apply_outline_time_ranges(summary, segments)
    assert summary.structured_outline[0].timestamp == "[00:00:00] - [00:03:40]"
