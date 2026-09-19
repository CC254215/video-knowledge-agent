from pathlib import Path

from app.config import Settings
from app.models import TranscriptSegment
from app.reasoning.conversation_agent import ConversationAgent
from app.transcript.multimodal_segmenter import build_multimodal_segments, load_multimodal_segments, save_multimodal_segments
from app.vision.frame_refiner import FrameEvidenceRefiner


def test_visual_low_agreement_triggers_refiner_once(monkeypatch, tmp_path: Path):
    segments = build_multimodal_segments(
        [TranscriptSegment(segment_id="seg_0000", start=0, end=10, text="这里可以看到一个演示步骤。")],
        [],
    )
    save_multimodal_segments(segments, tmp_path)
    calls = {"count": 0}
    original = FrameEvidenceRefiner.refine

    def fake_refine(self, video_id, question, target_time_range, segments, max_new_frames=10):
        calls["count"] += 1
        return original(self, video_id, question, target_time_range, segments, max_new_frames=max_new_frames)

    monkeypatch.setattr(FrameEvidenceRefiner, "refine", fake_refine)
    offline_settings = Settings(
        _env_file=None,
        openai_api_key=None,
        vision_api_key=None,
        llm_correlation_api_key=None,
        evidence_llm_api_key=None,
    )
    turn = ConversationAgent("v1", tmp_path, settings=offline_settings).answer("视频里这一步是怎么演示的？")

    assert calls["count"] == 1
    updated = load_multimodal_segments(tmp_path / "multimodal_segments.json")
    assert updated[0].representative_frame_ids
    assert turn.agreement_score < 0.3 or turn.confidence.value == "low"
