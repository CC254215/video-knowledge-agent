from pathlib import Path

from app.config import Settings
from app.models import FrameCaption, ModalityMode, ModalityProfile, MultimodalSegment, VideoMetadata
from app.reasoning.llm_client import LLMClient
from app.reasoning.summarizer import summarize_multimodal_video


def test_summary_prompt_includes_frame_caption(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_generate_json(self, prompt, schema_hint=None):
        captured["prompt"] = prompt
        return {
            "quick_overview": ["概览一", "概览二", "概览三"],
            "structured_outline": [
                {
                    "timestamp": "[00:00:00]",
                    "topic": "多模态证据",
                    "key_points": ["画面展示了 Agent Memory Pipeline"],
                    "segment_ids": ["seg_0000"],
                    "evidence_types": ["frame_caption"],
                }
            ],
            "deep_analysis": [],
            "action_items": [],
            "important_quotes": [],
            "open_questions": [],
            "evidence_coverage": {"covered_segment_ids": ["seg_0000"]},
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    segment = MultimodalSegment(
        segment_id="seg_0000",
        start=0,
        end=10,
        transcript_text="这里介绍流程。",
        representative_frame_ids=["f0"],
        visual_captions=[FrameCaption(frame_id="f0", timestamp=1, caption="画面展示 Agent Memory Pipeline。", model="fake")],
    )
    report = summarize_multimodal_video(
        VideoMetadata(video_id="v1", title="测试"),
        [segment],
        ModalityProfile(mode=ModalityMode.vision_supportive, speech_density=1.0, visual_required_score=0.4, ocr_required_score=0.0, reason="has frames"),
        Settings(_env_file=None, OPENAI_API_KEY="x", LLM_BASE_URL="https://example.test/v1", LLM_SUMMARY_MODEL="m"),
        tmp_path,
    )
    assert "Agent Memory Pipeline" in captured["prompt"]
    assert report.structured_outline[0].segment_ids == ["seg_0000"]
