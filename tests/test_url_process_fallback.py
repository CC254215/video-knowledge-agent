from pathlib import Path

from app.config import Settings
from app.models import VideoMetadata, VideoTranscript
from app import pipeline


def test_url_ingest_uses_subtitles_when_available(monkeypatch, tmp_path: Path):
    subtitle = tmp_path / "sub.vtt"
    subtitle.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n字幕内容\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "fetch_url_metadata", lambda url, output: (_metadata(), {"id": "url1"}))
    monkeypatch.setattr(pipeline, "download_video_file", lambda url, output: tmp_path / "video.mp4")
    monkeypatch.setattr(pipeline, "download_subtitle_file", lambda url, output: subtitle)

    called = {"asr": False}

    class FakeASR:
        def transcribe(self, media_path):
            called["asr"] = True
            return []

    monkeypatch.setattr(pipeline, "get_asr_adapter", lambda settings: FakeASR())
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"), STRICT_RUNTIME=False)
    video_id = pipeline.ingest_url("https://example.test/video", settings)
    transcript = (tmp_path / "data" / "videos" / video_id / "transcript.json").read_text(encoding="utf-8")
    assert "字幕内容" in transcript
    assert called["asr"] is False


def test_url_ingest_falls_back_to_asr_without_subtitles(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pipeline, "fetch_url_metadata", lambda url, output: (_metadata(), {"id": "url1"}))
    monkeypatch.setattr(pipeline, "download_video_file", lambda url, output: tmp_path / "video.mp4")
    monkeypatch.setattr(pipeline, "download_subtitle_file", lambda url, output: None)
    monkeypatch.setattr(pipeline, "download_audio_for_asr", lambda url, output: tmp_path / "source_audio.m4a")

    class FakeASR:
        def transcribe(self, media_path):
            return [{"start": 0, "end": 5, "text": "ASR 内容"}]

    monkeypatch.setattr(pipeline, "get_asr_adapter", lambda settings: FakeASR())
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"), STRICT_RUNTIME=False)
    video_id = pipeline.ingest_url("https://example.test/video", settings)
    transcript = (tmp_path / "data" / "videos" / video_id / "transcript.json").read_text(encoding="utf-8")
    assert "ASR 内容" in transcript


def test_url_ingest_reuses_cached_youtube_artifacts_before_metadata_fetch(monkeypatch, tmp_path: Path):
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"), STRICT_RUNTIME=True)
    out_dir = tmp_path / "data" / "videos" / "iGz2uWl-kGc"
    out_dir.mkdir(parents=True)
    metadata = VideoMetadata(
        video_id="iGz2uWl-kGc",
        title="Cached",
        source="Youtube",
        url="https://www.youtube.com/watch?v=iGz2uWl-kGc",
        duration=10,
    )
    pipeline.write_json(out_dir / "metadata.json", metadata)
    pipeline.write_json(out_dir / "transcript.json", VideoTranscript(video_id="iGz2uWl-kGc", segments=[], source="unknown"))

    def fail_metadata_fetch(url, output):
        raise AssertionError("metadata fetch should not run for cached YouTube video")

    monkeypatch.setattr(pipeline, "fetch_url_metadata", fail_metadata_fetch)

    video_id = pipeline.ingest_url("https://www.youtube.com/watch?v=iGz2uWl-kGc", settings)

    assert video_id == "iGz2uWl-kGc"


def _metadata() -> VideoMetadata:
    return VideoMetadata(video_id="url1", title="URL 视频", source="Youtube", url="https://example.test/video", duration=5)
