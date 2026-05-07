from app.config import Settings
from app.models import MultimodalSegment, RetrievedEvidence
from app.reasoning.llm_client import LLMClient
from app.reasoning.storyline import _merge_time_coverage_evidence, build_storyline_from_multimodal_segments


def test_storyline_node_without_evidence_becomes_unsupported(monkeypatch, tmp_path):
    def fake_generate_json(self, prompt, schema_hint=None):
        assert "OCR 已禁用" in prompt
        return {
            "nodes": [
                {
                    "node_id": "node_0000",
                    "time_start": 0,
                    "time_end": 10,
                    "topic": "无证据",
                    "claim": "没有引用证据",
                    "evidence_segment_ids": [],
                    "status": "supported",
                    "uncertainty": 0.1,
                }
            ]
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    settings = Settings(_env_file=None, OPENAI_API_KEY="x", OPENAI_BASE_URL="https://example.test/v1", LLM_MODEL="m")
    storyline = build_storyline_from_multimodal_segments("v1", [_segment()], settings=settings, log_dir=tmp_path)
    assert storyline.nodes[0].status.value == "unsupported"
    assert storyline.nodes[0].evidence_segment_ids == []


def test_storyline_supported_node_keeps_evidence(monkeypatch, tmp_path):
    def fake_generate_json(self, prompt, schema_hint=None):
        return {
            "nodes": [
                {
                    "node_id": "node_0000",
                    "time_start": 0,
                    "time_end": 10,
                    "topic": "推进",
                    "claim": "作者先提出核心问题。",
                    "evidence_segment_ids": ["seg_0000"],
                    "speech_evidence_ids": ["seg_0000"],
                    "modality_support": ["speech"],
                    "status": "supported",
                    "uncertainty": 0.2,
                }
            ]
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    settings = Settings(_env_file=None, OPENAI_API_KEY="x", OPENAI_BASE_URL="https://example.test/v1", LLM_MODEL="m")
    storyline = build_storyline_from_multimodal_segments("v1", [_segment()], settings=settings, log_dir=tmp_path)
    assert storyline.nodes[0].status.value == "supported"
    assert storyline.nodes[0].evidence_segment_ids == ["seg_0000"]


def _segment() -> MultimodalSegment:
    return MultimodalSegment(segment_id="seg_0000", start=0, end=10, transcript_text="作者先提出核心问题。")


def test_storyline_evidence_keeps_time_coverage_when_search_is_front_loaded():
    segments = [
        MultimodalSegment(segment_id=f"seg_{index:04d}", start=float(index * 100), end=float(index * 100 + 90), transcript_text=f"segment {index}")
        for index in range(6)
    ]
    front_loaded = [
        RetrievedEvidence(
            evidence_id="v1:seg_0000:speech",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="speech",
            text="front",
            start=0,
            end=90,
            score=1.0,
        )
    ]

    evidence = _merge_time_coverage_evidence("v1", segments, front_loaded, max_items=4)
    segment_ids = {item.segment_id for item in evidence}

    assert "seg_0000" in segment_ids
    assert "seg_0005" in segment_ids
