from app.config import Settings
from app.models import MultimodalSegment, RetrievedEvidence
from app.reasoning.llm_client import LLMClient
from app.reasoning.storyline import _backfill_storyline_gaps, _merge_time_coverage_evidence, _split_noncontiguous_storyline_nodes, build_storyline_from_multimodal_segments
from app.models import Storyline, StorylineNode, EvidenceStatus


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


def test_storyline_backfills_material_uncovered_time_gap():
    segments = [
        MultimodalSegment(segment_id="seg_0000", start=0, end=90, transcript_text="开场核心观点。"),
        MultimodalSegment(segment_id="seg_0001", start=90, end=190, transcript_text="被遗漏的事件一。"),
        MultimodalSegment(segment_id="seg_0002", start=190, end=290, transcript_text="被遗漏的事件二。"),
        MultimodalSegment(segment_id="seg_0003", start=290, end=390, transcript_text="被遗漏的事件三。"),
        MultimodalSegment(segment_id="seg_0004", start=390, end=480, transcript_text="结尾观点。"),
    ]
    storyline = Storyline(
        video_id="v1",
        query="q",
        nodes=[StorylineNode(
            node_id="node_0000",
            time_start=0,
            time_end=90,
            topic="开场",
            claim="开场核心观点。",
            summary="开场核心观点。",
            evidence_segment_ids=["seg_0000"],
            speech_evidence_ids=["seg_0000"],
            confidence=0.9,
            uncertainty=0.1,
            status=EvidenceStatus.supported,
        ), StorylineNode(
            node_id="node_0004",
            time_start=390,
            time_end=480,
            topic="结尾",
            claim="结尾观点。",
            summary="结尾观点。",
            evidence_segment_ids=["seg_0004"],
            speech_evidence_ids=["seg_0004"],
            confidence=0.9,
            uncertainty=0.1,
            status=EvidenceStatus.supported,
        )],
    )

    class FakeClient:
        def generate_json(self, prompt, schema_hint=None):
            return {
                "nodes": [{
                    "node_id": "gap_node",
                    "title": "缺失事件的模型总结",
                    "time_start": 90,
                    "time_end": 390,
                    "summary": "模型根据缺失时间段证据生成的总结。",
                    "key_points": ["总结要点"],
                    "evidence_segment_ids": ["seg_0001", "seg_0002", "seg_0003"],
                    "status": "supported",
                    "confidence": 0.9,
                }]
            }

    _backfill_storyline_gaps(storyline, segments, client=FakeClient(), metadata=None, query="q", max_gap_seconds=180)

    assert [node.node_id for node in storyline.nodes] == ["node_0000", "coverage_seg_0001_00", "node_0004"]
    assert storyline.nodes[1].evidence_segment_ids == ["seg_0001", "seg_0002", "seg_0003"]
    assert storyline.nodes[1].summary == "模型根据缺失时间段证据生成的总结。"
    assert any("coverage_gap_backfilled:seg_0001-seg_0003" in warning for warning in storyline.validation_warnings)


def test_storyline_split_removes_crossing_and_duplicate_evidence_ranges():
    segments = [
        MultimodalSegment(segment_id=f"seg_{index:04d}", start=index * 100, end=index * 100 + 90, transcript_text=f"事件{index}" )
        for index in range(4)
    ]
    def node(node_id, ids, start, end):
        return StorylineNode(
            node_id=node_id, time_start=start, time_end=end, topic=node_id, claim=node_id,
            summary=node_id, evidence_segment_ids=ids, speech_evidence_ids=ids,
            confidence=0.9, uncertainty=0.1, status=EvidenceStatus.supported,
        )
    storyline = Storyline(video_id="v1", query="q", nodes=[
        node("first", ["seg_0000", "seg_0001"], 0, 190),
        node("second", ["seg_0001", "seg_0002", "seg_0003"], 100, 390),
    ])

    _split_noncontiguous_storyline_nodes(storyline, segments)

    assert [(item.node_id, item.evidence_segment_ids, item.time_start, item.time_end) for item in storyline.nodes] == [
        ("first", ["seg_0000", "seg_0001"], 0, 190),
        ("second", ["seg_0002", "seg_0003"], 200, 390),
    ]
