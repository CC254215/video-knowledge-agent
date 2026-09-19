from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import VideoVisualProfile
from app.vision import video_classifier


@pytest.mark.parametrize("duration", [1200.0, 1.0, 0.01])
def test_probe_samples_cover_equal_intervals_of_entire_video(tmp_path, monkeypatch, duration):
    monkeypatch.setattr(video_classifier, "_probe_duration", lambda path: duration)
    monkeypatch.setattr(video_classifier, "_ffmpeg_executable", lambda: "ffmpeg")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"frame")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(video_classifier.subprocess, "run", run)
    stale = tmp_path / "probe_9999.jpg"
    stale.write_bytes(b"old")
    samples = video_classifier._sample_probe_frames(tmp_path / "video.mp4", tmp_path)
    timestamps = [float(command[command.index("-ss") + 1]) for command in commands]
    assert len(samples) == 12
    assert not stale.exists()
    assert timestamps == pytest.approx([duration * (i + 0.5) / 12 for i in range(12)], abs=1e-9)
    assert 0 < timestamps[0] < duration * 0.1
    assert duration * 0.9 < timestamps[-1] < duration
    assert all(command.index("-ss") < command.index("-i") for command in commands)


@pytest.mark.parametrize("payload, expected", [
    ({"streams": [{"duration": "12"}], "format": {"duration": "15"}}, 12.0),
    ({"streams": [{"duration": "N/A"}], "format": {"duration": "1200"}}, 1200.0),
    ({"format": {"duration": "nan"}}, None),
    ({"format": {"duration": "inf"}}, None),
    ({"format": {"duration": "0"}}, None),
])
def test_probe_duration_validates_stream_and_container_duration(monkeypatch, payload, expected):
    monkeypatch.setattr(video_classifier, "_ffprobe_executable", lambda: "ffprobe")
    monkeypatch.setattr(video_classifier.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    assert video_classifier._probe_duration(Path("video.mp4")) == expected


def test_failed_probe_frames_are_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(video_classifier, "_probe_duration", lambda path: 1200.0)
    monkeypatch.setattr(video_classifier, "_ffmpeg_executable", lambda: "ffmpeg")

    def run(command, **kwargs):
        target = Path(command[-1])
        if target.stem == "probe_0001":
            target.write_bytes(b"partial")
            return SimpleNamespace(returncode=1)
        if target.stem == "probe_0002":
            return SimpleNamespace(returncode=0)  # Decoder produced no frame.
        target.write_bytes(b"frame")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(video_classifier.subprocess, "run", run)
    samples = video_classifier._sample_probe_frames(tmp_path / "video.mp4", tmp_path, max_frames=4)
    assert [path.stem for path in samples] == ["probe_0000", "probe_0003"]


def test_unknown_duration_does_not_fall_back_to_opening_only(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(video_classifier, "_probe_video", lambda path: (1920, 1080, 30.0))
    monkeypatch.setattr(video_classifier, "_probe_duration", lambda path: None)
    monkeypatch.setattr(video_classifier.subprocess, "run", lambda *a, **kw: pytest.fail("must not extract without duration"))
    result = video_classifier.classify_video_visual_intensity("v1", video, tmp_path, settings=Settings(_env_file=None))
    assert result.reason == "insufficient_full_video_probe_frames"
    assert result.visual_class == "weak_visual"


def test_visual_classifier_classifies_weak_and_strong_from_lightweight_scores(tmp_path: Path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake-video")

    monkeypatch.setattr(video_classifier, "_probe_video", lambda path: (1920, 1080, 30.0))
    monkeypatch.setattr(video_classifier, "_sample_probe_frames", lambda path, output_dir: [tmp_path / "a.jpg", tmp_path / "b.jpg"])
    monkeypatch.setattr(video_classifier, "_compute_lightweight_scores", lambda paths: (0.9, 0.8, 0.7))

    settings = Settings(_env_file=None, visual_strength_threshold=0.48, strong_visual_frame_fps=1.5, weak_visual_frame_fps=0.2)
    strong = video_classifier.classify_video_visual_intensity("v1", video, tmp_path, has_subtitles=False, settings=settings)

    assert strong.visual_class == "strong_visual"
    assert strong.visual_strength_score >= settings.visual_strength_threshold
    assert strong.recommended_frame_fps == 1.5

    monkeypatch.setattr(video_classifier, "_compute_lightweight_scores", lambda paths: (0.01, 0.02, 0.02))
    weak = video_classifier.classify_video_visual_intensity("v1", video, tmp_path, has_subtitles=True, settings=settings)

    assert weak.visual_class == "weak_visual"
    assert weak.recommended_frame_fps == 0.2


def test_pipeline_uses_visual_profile_to_control_frame_fps(tmp_path: Path, monkeypatch):
    from app import pipeline
    from app.models import TranscriptSegment, VideoMetadata, VideoFrame

    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake-video")
    settings = Settings(_env_file=None, DATA_DIR=tmp_path / "data", FORCE_REFRESH=True)
    metadata = VideoMetadata(video_id="v1", title="demo", source="local", local_path=str(video))
    out_dir = pipeline.video_dir("v1", settings)
    pipeline.write_json(out_dir / "metadata.json", metadata)
    pipeline.write_json(
        out_dir / "transcript.json",
        {
            "video_id": "v1",
            "segments": [{"segment_id": "seg_0000", "start": 0.0, "end": 60.0, "text": "demo"}],
            "source": "asr",
        },
    )
    profile = VideoVisualProfile(
        video_id="v1",
        video_path=str(video),
        motion_score=0.9,
        visual_complexity_score=0.9,
        color_variation_score=0.9,
        visual_strength_score=0.9,
        visual_class="strong_visual",
        recommended_frame_fps=1.25,
        recommended_max_frames_per_segment=3,
        recommended_min_representative_frames=1,
        reason="test",
    )
    seen = {}

    monkeypatch.setattr(pipeline, "_classify_visual_profile", lambda *args, **kwargs: profile)

    def fake_extract(video_id, local_path, video_dir, fps=1.0):
        seen["fps"] = fps
        return [VideoFrame(frame_id="f0", timestamp=10.0, path=str(video), selected=True)]

    monkeypatch.setattr(pipeline, "extract_frames", fake_extract)
    monkeypatch.setattr(pipeline, "dedupe_frames", lambda frames: frames)
    monkeypatch.setattr(pipeline.VLMCaptioner, "caption_frames", lambda self, frames: [])

    segments = pipeline.ensure_multimodal_segments("v1", settings)

    assert seen["fps"] == 1.25
    assert segments[0].representative_frame_ids == ["f0"]
