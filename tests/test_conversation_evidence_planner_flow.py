from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment
from app.reasoning.conversation_agent import ConversationAgent
from app.reasoning.evidence_gap_planner import EvidenceAcquisitionPlan, EvidenceGapPlanner
from app.reasoning.evidence_refinement_tool import EvidenceRefinementToolResult, build_refinement_tool_call
from app.transcript.multimodal_segmenter import save_multimodal_segments


def test_conversation_agent_returns_no_refinement_when_planner_says_weak_related(monkeypatch, tmp_path: Path) -> None:
    save_multimodal_segments([MultimodalSegment(segment_id="seg_0", start=0, end=10, transcript_text="video topic")], tmp_path)
    monkeypatch.setattr(ConversationAgent, "_answer_once", lambda self, question, segments, refined: _turn(question, insufficient=True))
    monkeypatch.setattr(
        EvidenceGapPlanner,
        "decide",
        lambda self, question, metadata, segments, turn, evidence: EvidenceAcquisitionPlan(
            decision="no_refinement_needed",
            question_video_relevance="weak",
            evidence_sufficiency="not_applicable",
            reason="weakly related to outline",
        ),
    )

    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")).answer("outside?")

    assert "不做额外证据补充" in turn.answer
    assert "evidence_agent=no_refinement_needed" in turn.reason


def test_conversation_agent_executes_planner_tool_and_reanswers(monkeypatch, tmp_path: Path) -> None:
    save_multimodal_segments([MultimodalSegment(segment_id="seg_0", start=0, end=10, transcript_text="video topic")], tmp_path)
    calls = {"answer_once": 0, "tool": 0}

    def fake_answer_once(self, question, segments, refined):
        calls["answer_once"] += 1
        return _turn(question, insufficient=not refined)

    def fake_execute_tool(video_id, question, video_dir, segments, tool_call, settings):
        calls["tool"] += 1
        return EvidenceRefinementToolResult(status="updated", added_frames=1, added_captions=1)

    monkeypatch.setattr(ConversationAgent, "_answer_once", fake_answer_once)
    monkeypatch.setattr("app.evidence.orchestrator.execute_refinement_tool", fake_execute_tool)
    monkeypatch.setattr(
        EvidenceGapPlanner,
        "decide",
        lambda self, question, metadata, segments, turn, evidence: EvidenceAcquisitionPlan(
            decision="acquire_more_evidence",
            question_video_relevance="strong",
            evidence_sufficiency="insufficient",
            reason="needs more visual evidence",
            actions=[build_refinement_tool_call(question, query="video topic")],
        ),
    )

    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")).answer("topic?")

    assert calls == {"answer_once": 2, "tool": 1}
    assert turn.answer == "answered"
    assert "refinement_tool=partial_restart_for_evidence" in turn.reason


def test_conversation_agent_writes_trace_file(monkeypatch, tmp_path: Path) -> None:
    save_multimodal_segments([MultimodalSegment(segment_id="seg_0", start=0, end=10, transcript_text="video topic")], tmp_path)
    monkeypatch.setattr(ConversationAgent, "_answer_once", lambda self, question, segments, refined: _turn(question, insufficient=False))

    turn = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, CHROMA_PATH=tmp_path / "chroma")).answer("topic?")

    trace_files = list((tmp_path / "qa_traces").glob("*.json"))
    assert trace_files
    payload = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert payload["video_id"] == "v1"
    assert payload["question"] == "topic?"
    assert "trace_id=" in turn.reason
    assert {event["event"] for event in payload["events"]} >= {"question_received", "initial_turn", "final_turn"}


def _turn(question: str, insufficient: bool) -> ConversationTurn:
    return ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question=question,
        answer="当前视频证据不足以支持这个结论。" if insufficient else "answered",
        evidence=[] if insufficient else [{"segment_id": "seg_0", "evidence_type": "speech", "text": "video topic", "start": 0, "end": 10}],
        timestamps=[] if insufficient else ["00:00:00"],
        evidence_types=[] if insufficient else ["speech"],
        confidence=ConfidenceLevel.low if insufficient else ConfidenceLevel.medium,
        needs_visual_check=False,
        reason="test",
    )
