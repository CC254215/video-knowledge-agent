from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.models import ClaimRequirement, DeterministicQuestionContext, EvidencePlan, MultimodalSegment, QuestionAnalysis, RelationRequirement, ResolvedIntent, ScopeRequirement, VideoMetadata
from app.reasoning.evidence_requirement_planner import (
    build_evidence_planner_prompt,
    evidence_plan_schema,
    fallback_evidence_plan,
    plan_evidence,
)
from app.reasoning.question_analyzer import analyze_question, build_question_analyzer_prompt, question_analysis_schema
from app.reasoning.question_parser import SILENT_PLACEHOLDER, has_real_speech, parse_question_context


def resolve_intent(
    question: str,
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    settings: Settings | None = None,
    log_dir: Path | None = None,
    use_llm: bool = True,
) -> ResolvedIntent:
    """Coordinate question analysis and evidence planning, then expose the legacy view."""
    context = parse_question_context(question, metadata, segments)
    analysis = analyze_question(question, context, settings=settings, log_dir=log_dir, use_llm=use_llm)
    plan = plan_evidence(question, analysis, context, settings=settings, log_dir=log_dir, use_llm=use_llm)
    return adapt_to_legacy_intent(question, context, analysis, plan)


def adapt_to_legacy_intent(
    question: str,
    context: DeterministicQuestionContext,
    analysis: QuestionAnalysis,
    plan: EvidencePlan,
) -> ResolvedIntent:
    """One-way compatibility adapter; legacy fields never constrain the new models."""
    required_types = _modalities_to_evidence_types(plan.required_modalities)
    optional_types = [item for item in ("speech", "frame_caption") if item not in required_types]
    if not plan.claims:
        claims = [
            ClaimRequirement(
                id=f"c{index}", description=fact.fact, required=True,
                support_requirement="direct", preferred_modalities=[fact.modality], importance="core",
            )
            for index, fact in enumerate(plan.required_facts, start=1)
        ]
        claims.extend(
            ClaimRequirement(id=f"c{len(claims) + 1}", description=fact, required=True, importance="core")
            for fact in plan.decision_facts
            if fact and fact not in {item.description for item in claims}
        )
        plan = plan.model_copy(update={"claims": claims})
    if not plan.scope or plan.scope.target == "local" and plan.requires_global_coverage:
        strict_scope = any(token in question for token in ("完整", "全部", "所有", "一个都不要漏", "一个不能漏"))
        scope = ScopeRequirement(
            target="whole_video" if plan.requires_global_coverage else ("multi_event" if len(plan.required_facts) > 1 else "local"),
            completeness="strict" if strict_scope else ("best_effort" if plan.requires_global_coverage else "none"),
            allow_partial_answer=not strict_scope,
            description="由旧 EvidencePlan compatibility fields 迁移生成。",
        )
        plan = plan.model_copy(update={"scope": scope})
    return ResolvedIntent(
        original_question=question,
        rewritten_question=question,
        question_type=analysis.task_type,
        retrieval_queries=[],
        required_evidence_types=required_types,  # type: ignore[arg-type]
        optional_evidence_types=optional_types,  # type: ignore[arg-type]
        needs_visual_check="visual" in plan.required_modalities,
        can_use_general_knowledge=analysis.task_type in {"explanation", "critique", "action_items"},
        reason=f"question_analysis={analysis.reason}; evidence_plan={plan.reason}",
        temporal_scope=analysis.temporal_scope,
        temporal_relation=analysis.temporal_relation,
        evidence_requirements=plan.human_readable_requirements,
        distinguishing_facts=plan.decision_facts,
        min_distinct_visual_timestamps=(plan.min_distinct_timestamps if "visual" in plan.required_modalities else 0),
        resolution_method=f"question:{analysis.analysis_method};evidence:{plan.planning_method}",
        deterministic_context=context,
        question_analysis=analysis,
        evidence_plan=plan,
    )


def resolve_intent_rules(question: str, metadata: VideoMetadata, segments: list[MultimodalSegment]) -> ResolvedIntent:
    """Legacy entry point for deterministic context plus conservative fallbacks."""
    return resolve_intent(question, metadata, segments, use_llm=False)


def merge_llm_intent(
    question: str,
    fallback: ResolvedIntent,
    payload: dict[str, object],
    segments: list[MultimodalSegment] | None = None,
) -> ResolvedIntent:
    """Deprecated adapter for callers that already obtained QuestionAnalysis JSON."""
    context = fallback.deterministic_context or DeterministicQuestionContext(has_real_speech=has_real_speech(segments or []))
    try:
        analysis = QuestionAnalysis.model_validate({**payload, "analysis_method": "llm"})
    except ValidationError:
        return fallback
    # Parsed timestamps/ranges remain hard facts. Soft relation hints do not override this payload.
    if context.explicit_intervals:
        analysis = analysis.model_copy(update={"temporal_scope": "interval"})
    elif context.explicit_timestamps:
        analysis = analysis.model_copy(update={"temporal_scope": "point"})
    elif context.explicit_whole_video:
        analysis = analysis.model_copy(update={"temporal_scope": "whole_video"})
    return adapt_to_legacy_intent(question, context, analysis, fallback_evidence_plan(analysis, context))


def _modalities_to_evidence_types(modalities: list[str]) -> list[str]:
    output: list[str] = []
    if "speech" in modalities:
        output.append("speech")
    if "visual" in modalities:
        output.append("frame_caption")
    return output


__all__ = [
    "SILENT_PLACEHOLDER",
    "adapt_to_legacy_intent",
    "analyze_question",
    "build_evidence_planner_prompt",
    "build_question_analyzer_prompt",
    "evidence_plan_schema",
    "merge_llm_intent",
    "parse_question_context",
    "plan_evidence",
    "question_analysis_schema",
    "resolve_intent",
    "resolve_intent_rules",
]
