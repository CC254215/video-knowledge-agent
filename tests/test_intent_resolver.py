from __future__ import annotations

from app.models import MultimodalSegment, RetrievedEvidence, VideoMetadata
from app.reasoning.conversation_agent import build_conversation_prompt
from app.reasoning.intent_resolver import resolve_intent_rules


def test_action_advice_intent_rewrites_to_speech_action_items() -> None:
    metadata = VideoMetadata(video_id="v1", title="交易风险控制课程")
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=60,
            transcript_text="作者讲到交易曲线、风险控制、仓位管理和极端行情应对。",
        )
    ]

    intent = resolve_intent_rules("你能提供哪些操作上的建议？", metadata, segments)

    assert intent.question_type == "action_items"
    assert intent.needs_visual_check is False
    assert intent.required_evidence_types == ["speech"]
    assert "交易" in intent.rewritten_question
    assert intent.can_use_general_knowledge is True


def test_visual_operation_intent_requires_frame_caption() -> None:
    intent = resolve_intent_rules("视频里的操作界面按钮在哪里？", VideoMetadata(video_id="v1", title="软件教程"), [])

    assert intent.question_type == "visual_operation"
    assert intent.needs_visual_check is True
    assert "frame_caption" in intent.required_evidence_types


def test_prompt_contains_resolved_intent_and_general_knowledge_policy() -> None:
    intent = resolve_intent_rules(
        "你能提供哪些操作上的建议？",
        VideoMetadata(video_id="v1", title="交易风险控制课程"),
        [MultimodalSegment(segment_id="seg_0", start=0, end=60, transcript_text="风险控制")],
    )
    evidence = [
        RetrievedEvidence(
            evidence_id="v1:seg_0:speech",
            video_id="v1",
            segment_id="seg_0",
            evidence_type="speech",
            text="风险控制和仓位管理",
            start=0,
            end=60,
            score=0.9,
        )
    ]

    prompt = build_conversation_prompt(
        "你能提供哪些操作上的建议？",
        intent.question_type,
        evidence,
        history=[],
        needs_visual=intent.needs_visual_check,
        memory_context={"long_term_memory": [], "answer_policy": {}},
        resolved_intent=intent,
    )

    assert "resolved_intent" in prompt
    assert "action_items" in prompt
    assert "only_after_video_evidence_and_clearly_labeled" in prompt
