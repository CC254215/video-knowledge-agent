from __future__ import annotations

from app.models import DeterministicQuestionContext, EvidencePlan, QuestionAnalysis, RetrievalPlan, RetrievedEvidence


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def build_retrieval_plan(
    question: str,
    analysis: QuestionAnalysis,
    evidence_plan: EvidencePlan,
    context: DeterministicQuestionContext,
    existing_evidence: list[RetrievedEvidence] | None = None,
) -> RetrievalPlan:
    """Translate evidence facts into retrieval queries without changing question semantics."""
    existing_text = {item.text.strip().casefold() for item in existing_evidence or [] if item.text.strip()}
    candidates = [
        *[fact.fact for fact in evidence_plan.required_facts],
        *evidence_plan.decision_facts,
        analysis.answer_target,
        question,
    ]
    queries: list[str] = []
    prefer_chinese = _contains_cjk(question)
    for candidate in candidates:
        value = " ".join(candidate.split()).strip()
        if not value or value.casefold() in existing_text or value in queries:
            continue
        # Keep retrieval in the user's language. English LLM-generated facts
        # otherwise match English frame captions and systematically suppress
        # Chinese speech transcripts in the shared candidate pool.
        if prefer_chinese and not _contains_cjk(value):
            continue
        queries.append(value[:500])
        if len(queries) >= 8:
            break
    evidence_types: list[str] = []
    if "speech" in evidence_plan.required_modalities:
        evidence_types.append("speech")
    if "visual" in evidence_plan.required_modalities:
        evidence_types.extend(["frame_caption", "frame"])
    return RetrievalPlan(
        queries=queries or [question],
        evidence_types=evidence_types or ["speech", "frame_caption"],  # type: ignore[arg-type]
        time_constraints=[*context.explicit_intervals, *context.explicit_timestamps],
        reason="Queries come from required facts and decision facts; intent analysis does not generate retrieval queries.",
    )
