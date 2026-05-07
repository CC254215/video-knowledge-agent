from pathlib import Path

from app.config import Settings
from app.models import ModalityMode, ModalityProfile, MultimodalSegment, VideoMetadata
from app.reasoning.llm_client import LLMClient
from app.reasoning.summarizer import summarize_multimodal_video


def test_summary_key_points_keep_segment_ids(monkeypatch, tmp_path: Path):
    def fake_generate_json(self, prompt, schema_hint=None):
        assert "OCR 已禁用" in prompt
        return {
            "quick_overview": ["一句", "二句", "三句"],
            "structured_outline": [{"timestamp": "[00:00:00]", "topic": "论点", "key_points": ["必须绑定证据"], "segment_ids": ["seg_0000"]}],
            "deep_analysis": [{"claim": "必须绑定证据", "evidence_segment_ids": ["seg_0000"], "caveats": []}],
            "action_items": [],
            "important_quotes": [],
            "open_questions": [],
            "evidence_coverage": {"covered_segment_ids": ["seg_0000"]},
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    report = summarize_multimodal_video(_metadata(), [_segment()], _profile(), _settings(), tmp_path)
    assert report.structured_outline[0].segment_ids == ["seg_0000"]
    assert report.warnings == []


def test_summary_invalid_segment_id_is_warning(monkeypatch, tmp_path: Path):
    def fake_generate_json(self, prompt, schema_hint=None):
        return {
            "quick_overview": ["一句"],
            "structured_outline": [{"timestamp": "[00:00:00]", "topic": "坏引用", "key_points": ["x"], "segment_ids": ["missing"]}],
            "deep_analysis": [],
            "action_items": [],
            "important_quotes": [],
            "open_questions": [],
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    report = summarize_multimodal_video(_metadata(), [_segment()], _profile(), _settings(), tmp_path)
    assert any("invalid_segment_id" in warning for warning in report.warnings)
    assert any("evidence_missing" in warning for warning in report.warnings)
    assert report.confidence.value == "low"


def test_summary_outline_tail_is_patched_when_llm_stops_early(monkeypatch, tmp_path: Path):
    def fake_generate_json(self, prompt, schema_hint=None):
        return {
            "quick_overview": ["overview"],
            "structured_outline": [{"timestamp": "[00:00:00]", "topic": "front", "key_points": ["front"], "segment_ids": ["seg_0000"]}],
            "deep_analysis": [],
            "action_items": [],
            "important_quotes": [],
            "open_questions": [],
            "evidence_coverage": {"covered_segment_ids": ["seg_0000"]},
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    segments = [
        MultimodalSegment(segment_id=f"seg_{index:04d}", start=float(index * 100), end=float(index * 100 + 90), transcript_text=f"segment {index}")
        for index in range(6)
    ]

    report = summarize_multimodal_video(_metadata(), segments, _profile(), _settings(), tmp_path)

    covered = {segment_id for item in report.structured_outline for segment_id in item.segment_ids}
    assert "seg_0005" in covered
    assert any("outline_time_coverage_patched" in warning for warning in report.warnings)


def _metadata() -> VideoMetadata:
    return VideoMetadata(video_id="v1", title="测试视频")


def _segment() -> MultimodalSegment:
    return MultimodalSegment(segment_id="seg_0000", start=0, end=10, transcript_text="视频说结论必须有证据。")


def _profile() -> ModalityProfile:
    return ModalityProfile(mode=ModalityMode.text_dominant, speech_density=1.0, visual_required_score=0.0, ocr_required_score=0.0, reason="text")


def _settings() -> Settings:
    return Settings(_env_file=None, OPENAI_API_KEY="x", OPENAI_BASE_URL="https://example.test/v1", LLM_MODEL="m")
