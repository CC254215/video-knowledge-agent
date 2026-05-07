from pathlib import Path

from app.config import Settings
from app.models import VideoVisualProfile
from app.vision import video_classifier


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
