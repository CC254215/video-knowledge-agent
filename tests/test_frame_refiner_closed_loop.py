from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, TranscriptSegment
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.transcript.multimodal_segmenter import build_multimodal_segments
from app.vision.frame_refiner import FrameEvidenceRefiner
from app.vision.vlm_captioner import VLMCaptioner


def test_frame_refiner_updates_segments_and_chroma(tmp_path: Path, monkeypatch):
    segments = build_multimodal_segments([TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里可以看到演示步骤。")], [])

    def fake_caption_frames(self, frames):
        return [
            FrameCaption(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                image_path=frame.path,
                caption="补帧画面展示了一个演示步骤。",
                model="fake-vlm",
                status="success",
            )
            for frame in frames
        ]

    monkeypatch.setattr(VLMCaptioner, "caption_frames", fake_caption_frames)
    refined = FrameEvidenceRefiner(tmp_path, settings=Settings(_env_file=None)).refine("v1", "视频里这个步骤是怎么演示的？", (0, 10), segments)
    assert refined[0].representative_frame_ids
    assert refined[0].visual_captions
    assert "frame_caption" in refined[0].evidence_types

    settings = Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")
    store = ChromaMemoryStore(settings)
    store.add_multimodal_segments("v1", refined)
    results = store.search("v1", "演示步骤", evidence_types=["frame_caption"])
    assert results
    assert results[0].evidence_type == "frame_caption"
