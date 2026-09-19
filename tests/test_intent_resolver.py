from __future__ import annotations

from app.config import Settings
from app.models import EvidencePlan, MultimodalSegment, QuestionAnalysis, RequiredFact, RetrievedEvidence, VideoMetadata
from app.reasoning.conversation_agent import _evidence_contract_gaps, _is_silent_placeholder, build_conversation_prompt
from app.reasoning.intent_resolver import (
    adapt_to_legacy_intent,
    analyze_question,
    build_evidence_planner_prompt,
    build_question_analyzer_prompt,
    parse_question_context,
    plan_evidence,
    resolve_intent_rules,
)
from app.reasoning.llm_client import LLMClient
from app.reasoning.retrieval_planner import build_retrieval_plan


def _metadata() -> VideoMetadata:
    return VideoMetadata(video_id="v1", title="Demo", duration=180)


def _speech_segments() -> list[MultimodalSegment]:
    return [MultimodalSegment(segment_id="seg_0", start=0, end=180, transcript_text="老师先介绍数据集，然后介绍模型结构。")]


def _silent_segments() -> list[MultimodalSegment]:
    return [MultimodalSegment(segment_id="seg_0", start=0, end=180, transcript_text="Silent video. No spoken transcript is available; use visual evidence.")]


def _llm_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        LLM_API_KEY="test-key",
        LLM_BASE_URL="https://example.invalid/v1",
        LLM_MODEL="test-model",
        CHROMA_PATH=tmp_path / "chroma",
    )


def test_explicit_timestamp_is_a_hard_point_but_plain_colons_are_not() -> None:
    point = parse_question_context("01:20 时老师在讲什么？", _metadata(), _speech_segments())
    assert [(item.start, item.source_text) for item in point.explicit_timestamps] == [(80.0, "01:20")]
    intent = resolve_intent_rules("01:20 时老师在讲什么？", _metadata(), _speech_segments())
    assert intent.question_analysis.temporal_scope == "point"

    choices = parse_question_context("Choices:\nA. first\nB. second\nAnswer:", _metadata(), [])
    assert choices.explicit_timestamps == []
    assert choices.choice_based is True


def test_explicit_interval_and_whole_video_are_separate_scopes() -> None:
    interval = resolve_intent_rules("从 01:20 到 02:10 发生了什么？", _metadata(), [])
    whole = resolve_intent_rules("总结整个视频的内容。", _metadata(), _speech_segments())
    assert interval.question_analysis.temporal_scope == "interval"
    assert interval.deterministic_context.explicit_intervals[0].start == 80
    assert whole.question_analysis.temporal_scope == "whole_video"


def test_process_word_does_not_mechanically_mean_sequence() -> None:
    intent = resolve_intent_rules("Explain the process of backpropagation.", _metadata(), _speech_segments())
    assert intent.question_analysis.task_type == "explanation"
    assert intent.question_analysis.temporal_relation == "none"


def test_speech_and_visual_sequences_do_not_share_a_forced_modality() -> None:
    speech = resolve_intent_rules("老师是先介绍数据集还是先介绍模型结构？", _metadata(), _speech_segments())
    visual = resolve_intent_rules("这个人先拿起木板还是先用手塑形？", _metadata(), [])
    assert speech.question_analysis.temporal_relation == "sequence"
    assert speech.question_analysis.content_modalities == ["speech"]
    assert speech.evidence_plan.required_modalities == ["speech"]
    assert visual.question_analysis.temporal_relation == "sequence"
    assert visual.question_analysis.content_modalities == ["visual"]
    assert visual.evidence_plan.required_modalities == ["visual"]


def test_before_after_is_event_local_and_stage_can_be_whole_video() -> None:
    local = resolve_intent_rules("在打开设置之前，老师做了什么？", _metadata(), _speech_segments())
    stage = resolve_intent_rules("整个视频中的制作过程可以分为哪些阶段？", _metadata(), _speech_segments())
    assert (local.question_analysis.temporal_scope, local.question_analysis.temporal_relation) == ("event_local", "before_after")
    assert (stage.question_analysis.temporal_scope, stage.question_analysis.temporal_relation) == ("whole_video", "stage")


def test_silent_placeholder_is_metadata_not_speech_evidence() -> None:
    context = parse_question_context("视频里发生了什么？", _metadata(), _silent_segments())
    assert context.has_real_speech is False
    assert context.has_audio is False
    assert "Silent video" not in context.transcript_preview
    intent = resolve_intent_rules("视频里发生了什么？", _metadata(), _silent_segments())
    assert intent.evidence_plan.required_modalities == ["visual"]

    marker = RetrievedEvidence(evidence_id="silent", video_id="v1", segment_id="seg_0", evidence_type="speech", text="Silent video. No spoken transcript is available; use visual evidence.", start=0, end=180)
    assert _is_silent_placeholder(marker)


def test_explicit_speech_question_remains_speech_requirement_when_silent() -> None:
    intent = resolve_intent_rules("老师说了什么？", _metadata(), _silent_segments())
    assert intent.question_analysis.content_modalities == ["speech"]
    assert intent.evidence_plan.required_modalities == ["speech"]
    marker = RetrievedEvidence(evidence_id="silent", video_id="v1", segment_id="seg_0", evidence_type="speech", text="Silent video. No spoken transcript is available; use visual evidence.", start=0, end=180)
    assert "missing_modality:speech" in _evidence_contract_gaps(intent, [marker])


def test_llm_can_correct_soft_relation_hint_but_not_explicit_timestamp(monkeypatch, tmp_path) -> None:
    responses = [
        {
            "task_type": "explanation", "temporal_scope": "none", "temporal_relation": "none",
            "answer_target": "the conceptual consequence", "content_modalities": ["speech"],
            "choice_based": False, "reason": "after is used conceptually here",
        },
        {
            "task_type": "factual", "temporal_scope": "none", "temporal_relation": "none",
            "answer_target": "what happens at the specified time", "content_modalities": ["visual"],
            "choice_based": False, "reason": "point observation",
        },
    ]
    monkeypatch.setattr(LLMClient, "generate_json", lambda self, prompt, schema_hint=None: responses.pop(0))
    settings = _llm_settings(tmp_path)

    soft_context = parse_question_context("Explain what changes after normalization in this concept.", _metadata(), _speech_segments())
    assert soft_context.rule_hints.possible_temporal_relation == "before_after"
    corrected = analyze_question("Explain what changes after normalization in this concept.", soft_context, settings=settings)
    assert corrected.temporal_relation == "none"

    point_context = parse_question_context("What is happening at 01:20?", _metadata(), [])
    point = analyze_question("What is happening at 01:20?", point_context, settings=settings)
    assert point.temporal_scope == "point"


def test_multimodal_analysis_and_evidence_planner_are_two_calls(monkeypatch, tmp_path) -> None:
    responses = [
        {
            "task_type": "factual", "temporal_scope": "event_local", "temporal_relation": "none",
            "answer_target": "the figure shown when loss is introduced", "content_modalities": ["speech", "visual"],
            "choice_based": False, "reason": "align spoken event with screen content",
        },
        {
            "required_modalities": ["speech", "visual"],
            "required_facts": [
                {"fact": "locate when the loss function is introduced", "modality": "speech"},
                {"fact": "identify the figure visible at that time", "modality": "visual"},
            ],
            "decision_facts": ["the identity of the displayed figure"],
            "min_distinct_timestamps": 1, "requires_temporal_order": False,
            "requires_global_coverage": False, "requires_visual_identity": True,
            "human_readable_requirements": ["align the spoken event and displayed figure"],
            "reason": "both modalities are necessary",
        },
    ]
    prompts: list[str] = []
    def fake_generate(self, prompt, schema_hint=None):
        prompts.append(prompt)
        return responses.pop(0)
    monkeypatch.setattr(LLMClient, "generate_json", fake_generate)
    settings = _llm_settings(tmp_path)
    question = "老师介绍 loss 函数时，屏幕上展示了什么图？"
    context = parse_question_context(question, _metadata(), _speech_segments())
    analysis = analyze_question(question, context, settings=settings)
    plan = plan_evidence(question, analysis, context, settings=settings)
    assert analysis.content_modalities == ["speech", "visual"]
    assert plan.required_modalities == ["speech", "visual"]
    assert len(prompts) == 2
    assert "不要搜索证据" in prompts[0]
    assert "required_facts" in prompts[1]
    assert "规则基线" not in prompts[0]


def test_enum_validation_rejects_invalid_llm_ontology(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(LLMClient, "generate_json", lambda self, prompt, schema_hint=None: {
        "task_type": "time_navigation", "temporal_scope": "sequence", "temporal_relation": "visual_operation",
        "answer_target": "bad", "content_modalities": ["frames"], "choice_based": False, "reason": "bad",
    })
    context = parse_question_context("Explain the topic.", _metadata(), _speech_segments())
    analysis = analyze_question("Explain the topic.", context, settings=_llm_settings(tmp_path))
    assert analysis.analysis_method == "fallback"
    assert analysis.task_type == "explanation"


def test_temporal_contract_counts_valid_events_in_required_modality() -> None:
    context = parse_question_context("老师先讲哪个？", _metadata(), _speech_segments())
    analysis = QuestionAnalysis(task_type="factual", temporal_scope="whole_video", temporal_relation="sequence", answer_target="topic order", content_modalities=["speech"])
    plan = EvidencePlan(
        required_modalities=["speech"], required_facts=[RequiredFact(fact="identify both spoken topics", modality="speech")],
        decision_facts=["which topic occurs first"], min_distinct_timestamps=2, requires_temporal_order=True,
    )
    intent = adapt_to_legacy_intent("老师先讲哪个？", context, analysis, plan)
    first = RetrievedEvidence(evidence_id="s1", video_id="v1", segment_id="a", evidence_type="speech", text="先介绍数据集", start=20, end=25, timestamp=20)
    second = RetrievedEvidence(evidence_id="s2", video_id="v1", segment_id="b", evidence_type="speech", text="接下来介绍优化器", start=80, end=85, timestamp=80)
    assert _evidence_contract_gaps(intent, [first]) == ["distinct_timestamps:1/2"]
    assert _evidence_contract_gaps(intent, [first, second]) == []


def test_retrieval_planner_uses_required_and_decision_facts() -> None:
    context = parse_question_context("Which method occurs first?", _metadata(), [])
    analysis = QuestionAnalysis(task_type="procedure", temporal_scope="whole_video", temporal_relation="sequence", answer_target="shaping process", content_modalities=["visual"])
    plan = EvidencePlan(
        required_modalities=["visual"],
        required_facts=[RequiredFact(fact="identify the rough shaping method", modality="visual")],
        decision_facts=["whether hand shaping occurs before board shaping"], min_distinct_timestamps=2,
    )
    retrieval = build_retrieval_plan("Which method occurs first?", analysis, plan, context)
    assert retrieval.queries[:2] == ["identify the rough shaping method", "whether hand shaping occurs before board shaping"]
    assert retrieval.evidence_types == ["frame_caption", "frame"]


def test_chinese_retrieval_plan_drops_english_llm_fact_rewrites() -> None:
    context = parse_question_context("老师展示了哪些案例？", _metadata(), [])
    analysis = QuestionAnalysis(
        task_type="factual", temporal_scope="whole_video", answer_target="the cases demonstrated", content_modalities=["speech"],
    )
    plan = EvidencePlan(
        required_modalities=["speech"],
        required_facts=[RequiredFact(fact="The teacher demonstrates several cases", modality="speech")],
        decision_facts=["the complete list of cases"],
    )
    retrieval = build_retrieval_plan("老师展示了哪些案例？", analysis, plan, context)
    assert retrieval.queries == ["老师展示了哪些案例？"]


def test_legacy_view_keeps_existing_prompt_operational() -> None:
    intent = resolve_intent_rules("请解释老师提到的风险控制。", VideoMetadata(video_id="v1", title="课程"), _speech_segments())
    evidence = [RetrievedEvidence(evidence_id="v1:s:speech", video_id="v1", segment_id="s", evidence_type="speech", text="风险控制", start=0, end=10, score=0.9)]
    prompt = build_conversation_prompt(
        intent.original_question, intent.question_type, evidence, history=[], needs_visual=intent.needs_visual_check,
        memory_context={"long_term_memory": [], "answer_policy": {}}, resolved_intent=intent,
    )
    assert "resolved_intent" in prompt
    assert '"question_analysis"' in prompt
    assert intent.question_type == "explanation"


def test_prompts_have_separate_responsibilities() -> None:
    context = parse_question_context("What happens at 01:20?", _metadata(), [])
    analysis_prompt = build_question_analyzer_prompt("What happens at 01:20?", context)
    analysis = QuestionAnalysis(answer_target="the event at 01:20", content_modalities=["visual"], temporal_scope="point")
    evidence_prompt = build_evidence_planner_prompt("What happens at 01:20?", analysis, context)
    assert "不要生成 retrieval query" in analysis_prompt
    assert "required_facts" not in analysis_prompt
    assert "required_facts" in evidence_prompt
    assert "不要生成检索查询" in evidence_prompt
