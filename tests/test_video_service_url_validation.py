from pathlib import Path

from app.config import Settings
from app.models import ModalityMode, ModalityProfile, Storyline, SummaryReport, VideoMetadata
from app.services.video_service import ProcessVideoResult, VideoService


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
