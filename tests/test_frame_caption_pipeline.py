from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, TranscriptSegment, VideoFrame
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.transcript.multimodal_segmenter import build_multimodal_segments, load_multimodal_segments, save_multimodal_segments
from app.vision.vlm_captioner import VLMCaptioner


def test_frame_caption_written_to_segments_and_chroma(tmp_path: Path, monkeypatch):
    image = tmp_path / "f0.jpg"
    image.write_bytes(b"fake-image")
    frame = VideoFrame(frame_id="f0", timestamp=5, path=str(image), selected=True)

    def fake_caption_frame(self, image_path: str, context: str | None = None):
        return FrameCaption(
            frame_id="f0",
            timestamp=5,
            image_path=image_path,
            caption="画面展示了 Agent Memory Pipeline 的流程图。",
            model="fake-vlm",
            status="success",
        )

    monkeypatch.setattr(VLMCaptioner, "caption_frame", fake_caption_frame)
    caption = VLMCaptioner(Settings(_env_file=None)).caption_frame(str(image), context="agent memory")
    segments = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里介绍 agent memory。")],
        [frame],
        {"seg_0000": ["f0"]},
        [caption],
    )
    save_multimodal_segments(segments, tmp_path)
    loaded = load_multimodal_segments(tmp_path / "multimodal_segments.json")

    assert loaded[0].visual_captions[0].caption
    assert loaded[0].frame_caption_status == "success"
    assert "frame_caption" in loaded[0].evidence_types

    settings = Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")
    store = ChromaMemoryStore(settings)
    store.add_multimodal_segments("v1", loaded)
    results = store.search("v1", "Agent Memory Pipeline 流程图", evidence_types=["frame_caption"])
    assert results
    assert results[0].evidence_type == "frame_caption"


def test_vlm_caption_frames_preserves_order_with_bounded_concurrency(tmp_path: Path, monkeypatch):
    frames = []
    for index in range(4):
        image = tmp_path / f"f{index}.jpg"
        image.write_bytes(b"fake-image")
        frames.append(VideoFrame(frame_id=f"f{index}", timestamp=float(index), path=str(image), selected=True))

    settings = Settings(
        _env_file=None,
        vision_api_key="key",
        vlm_base_url="https://example.test/v4",
        vlm_caption_model="vision-test",
        vlm_caption_concurrency=2,
        vlm_caption_retries=0,
    )

    def fake_caption_one(self, frame: VideoFrame, context: str | None = None):
        return FrameCaption(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            image_path=frame.path,
            caption=f"caption-{frame.frame_id}",
            model="vision-test",
            status="success",
        )

    monkeypatch.setattr(VLMCaptioner, "_caption_one", fake_caption_one)
    captions = VLMCaptioner(settings).caption_frames(frames)

    assert [caption.frame_id for caption in captions] == [frame.frame_id for frame in frames]
    assert all(caption.status == "success" for caption in captions)
