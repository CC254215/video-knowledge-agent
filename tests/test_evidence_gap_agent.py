from __future__ import annotations

from app.models import ConfidenceLevel, EvidenceGapDecision, MultimodalSegment, RetrievedEvidence, VideoMetadata
from app.modality.router import question_needs_visual_check
from app.reasoning.conversation_agent import classify_question
from app.reasoning.evidence_gap_agent import EvidenceGapAgent, extract_time_anchors


def test_time_anchor_extracts_ten_minutes() -> None:
    anchors = extract_time_anchors("作者在十分钟的时候提了哪些问题？", duration=1200)
    assert anchors
    assert anchors[0].timestamp == 600


def test_gap_agent_refines_missing_time_anchor() -> None:
    metadata = VideoMetadata(video_id="v1", title="交易课程", duration=1200)
    segments = [MultimodalSegment(segment_id="seg_0", start=0, end=120, transcript_text="开头介绍交易曲线")]
    evidence = [
        RetrievedEvidence(
            evidence_id="v1:seg_0:speech",
            video_id="v1",
            segment_id="seg_0",
            evidence_type="speech",
            text="开头介绍交易曲线",
            start=0,
            end=120,
            score=0.8,
        )
    ]

    decision = EvidenceGapAgent().decide(
        "作者在十分钟的时候提了哪些问题？",
        metadata,
        segments,
        evidence,
        agreement=0.7,
        confidence=ConfidenceLevel.low.value,
        needs_visual=False,
    )

    assert decision.is_video_relevant is True
    assert decision.should_refine is True
    assert decision.target_ranges[0].start == 540
    assert decision.target_ranges[0].end == 660
    assert any("missing_time_anchor" in reason for reason in decision.reasons)


def test_action_advice_does_not_force_visual_check() -> None:
    assert classify_question("你能提供哪些操作上的建议？") == "action_items"
    assert question_needs_visual_check("你能提供哪些操作上的建议？") is False
    assert question_needs_visual_check("视频里的操作界面按钮在哪里？") is True


def test_unrelated_question_can_early_exit() -> None:
    metadata = VideoMetadata(video_id="v1", title="气球物理实验", duration=120)
    decision = EvidenceGapAgent().decide(
        "你好",
        metadata,
        [],
        [],
        agreement=0,
        confidence="low",
        needs_visual=False,
    )

    assert isinstance(decision, EvidenceGapDecision)
    assert decision.is_video_relevant is False
    assert decision.early_exit_reply
