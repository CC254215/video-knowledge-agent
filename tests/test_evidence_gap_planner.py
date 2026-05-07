from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment, VideoMetadata
from app.reasoning.evidence_gap_planner import EvidenceGapPlanner
from app.reasoning.llm_client import LLMClient


def test_evidence_gap_planner_uses_llm_no_refinement(monkeypatch, tmp_path: Path) -> None:
    def fake_generate_json(self, prompt, schema_hint=None):
        assert "video_outline" in prompt
        assert "partial_restart_for_evidence" in prompt
        return {
            "decision": "no_refinement_needed",
            "question_video_relevance": "weak",
            "evidence_sufficiency": "not_applicable",
            "reason": "The question is weakly related to the outline.",
            "actions": [],
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    planner = EvidenceGapPlanner(_evidence_settings(tmp_path), tmp_path / "llm_logs")

    plan = planner.decide(
        "unrelated question",
        VideoMetadata(video_id="v1", title="Video"),
        [MultimodalSegment(segment_id="seg_0", start=0, end=10, transcript_text="video topic")],
        _insufficient_turn("unrelated question"),
        [],
    )

    assert plan.decision == "no_refinement_needed"
    assert plan.question_video_relevance == "weak"
    assert not plan.actions


def test_evidence_gap_planner_coerces_llm_tool_action(monkeypatch, tmp_path: Path) -> None:
    def fake_generate_json(self, prompt, schema_hint=None):
        return {
            "decision": "acquire_more_evidence",
            "question_video_relevance": "strong",
            "evidence_sufficiency": "insufficient",
            "reason": "Needs visual evidence around the related text.",
            "actions": [
                {
                    "tool": "partial_restart_for_evidence",
                    "arguments": {
                        "query": "related text",
                        "target_ranges": [],
                        "top_k_text_segments": 2,
                        "context_padding_seconds": 5,
                    },
                }
            ],
        }

    monkeypatch.setattr(LLMClient, "generate_json", fake_generate_json)
    planner = EvidenceGapPlanner(_evidence_settings(tmp_path), tmp_path / "llm_logs")

    plan = planner.decide(
        "related text?",
        VideoMetadata(video_id="v1", title="Video"),
        [MultimodalSegment(segment_id="seg_0", start=0, end=10, transcript_text="related text")],
        _insufficient_turn("related text?"),
        [],
    )

    assert plan.should_refine
    assert plan.actions[0].tool == "partial_restart_for_evidence"
    assert plan.actions[0].arguments.query == "related text"
    assert plan.actions[0].arguments.top_k_text_segments == 2


def _evidence_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        CHROMA_PATH=tmp_path / "chroma",
        evidence_llm_api_key="test-key",
        evidence_llm_base_url="https://example.test/api/paas/v4",
        evidence_llm_model="glm-5.1",
    )


def _insufficient_turn(question: str) -> ConversationTurn:
    return ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question=question,
        answer="当前视频证据不足以支持这个结论。",
        evidence=[],
        timestamps=[],
        evidence_types=[],
        confidence=ConfidenceLevel.low,
        needs_visual_check=False,
        reason="test",
    )
