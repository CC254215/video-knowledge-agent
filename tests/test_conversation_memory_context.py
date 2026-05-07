from __future__ import annotations

from app.models import RetrievedEvidence
from app.reasoning.conversation_agent import build_conversation_prompt, compose_grounded_answer


def test_conversation_prompt_keeps_mempalace_separate_from_video_evidence() -> None:
    evidence = [
        RetrievedEvidence(
            evidence_id="v1:seg_0000:speech",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="speech",
            text="视频中作者解释了气球压力实验。",
            start=0,
            end=60,
            score=0.9,
        )
    ]
    memory_context = {
        "long_term_memory": [
            {
                "source": "mempalace",
                "memory_type": "user_preference",
                "content": "用户偏好简洁回答。",
                "usage_rule": "May guide answer style, not factual video claims.",
            }
        ],
        "answer_policy": {"video_claims_require_video_evidence": True},
    }

    prompt = build_conversation_prompt(
        "作者为什么这样解释？",
        "explanation",
        evidence,
        history=[],
        needs_visual=False,
        memory_context=memory_context,
    )

    assert "long_term_memory_context" in prompt
    assert "用户偏好简洁回答" in prompt
    assert "不能替代当前视频证据" in prompt
    assert "v1:seg_0000:speech" in prompt


def test_conversation_prompt_requires_synthesis_before_background() -> None:
    evidence = [
        RetrievedEvidence(
            evidence_id="v1:seg_0000:speech",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="speech",
            text="视频证据说明，作者先给出结论，再解释原因。",
            start=0,
            end=60,
            score=0.9,
        )
    ]

    prompt = build_conversation_prompt(
        "怎么理解作者的观点？",
        "explanation",
        evidence,
        history=[],
        needs_visual=False,
        memory_context={"long_term_memory": [], "answer_policy": {}},
    )

    assert "不是证据摘录器" in prompt
    assert "不要把 evidence 文本原样堆出来" in prompt
    assert "背景知识补充" in prompt
    assert "only_after_video_evidence_and_clearly_labeled" in prompt


def test_local_grounded_answer_synthesizes_instead_of_dumping_evidence() -> None:
    evidence = [
        RetrievedEvidence(
            evidence_id="v1:seg_0000:speech",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="speech",
            text="作者认为需要先确认事实，再做行动判断。",
            start=0,
            end=60,
            score=0.9,
        )
    ]

    answer = compose_grounded_answer("explanation", evidence)

    assert answer.startswith("根据相关片段，可以理解为")
    assert "关键依据" in answer
    assert "v1:seg_0000:speech" in answer
