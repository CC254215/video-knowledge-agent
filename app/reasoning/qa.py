from __future__ import annotations

from app.models import ConfidenceLevel, QAAnswer, TranscriptSegment
from app.modality.router import question_needs_visual_check
from app.reasoning.summarizer import format_timestamp
from app.retrieval.dpp import greedy_dpp_select
from app.retrieval.evidence import EvidenceRetriever, compute_evidence_agreement, confidence_from_agreement
from app.retrieval.vector_store import InMemoryVectorStore


def answer_question(
    video_id: str,
    question: str,
    store: InMemoryVectorStore,
    top_k: int = 6,
) -> QAAnswer:
    retriever = EvidenceRetriever(store)
    query_evidence = retriever.retrieve_query_evidence(video_id, question, top_k=top_k)
    segments = store.get_segments(video_id, query_evidence.segment_ids)
    if not segments:
        return QAAnswer(
            question=question,
            answer="当前视频证据不足，无法回答这个问题。",
            evidence_segment_ids=[],
            timestamps=[],
            quotes=[],
            confidence=ConfidenceLevel.low,
            needs_visual_check=question_needs_visual_check(question),
            reason="没有检索到相关 transcript segment。",
        )

    candidate_answer = _compose_candidate_answer(segments)
    answer_evidence = retriever.retrieve_answer_evidence(video_id, candidate_answer, top_k=top_k)
    query_results = store.search(video_id, question, top_k=top_k)
    answer_results = store.search(video_id, candidate_answer, top_k=top_k)
    query_selected = greedy_dpp_select(
        [result.segment.segment_id for result in query_results],
        [result.score for result in query_results],
        [result.embedding for result in query_results],
        k=min(3, len(query_results)),
    )
    answer_selected = greedy_dpp_select(
        [result.segment.segment_id for result in answer_results],
        [result.score for result in answer_results],
        [result.embedding for result in answer_results],
        k=min(3, len(answer_results)),
    )
    agreement = compute_evidence_agreement(query_selected, answer_selected)
    confidence = ConfidenceLevel(confidence_from_agreement(agreement))
    final_ids = query_selected or query_evidence.segment_ids[:3]
    final_segments = store.get_segments(video_id, final_ids)
    return QAAnswer(
        question=question,
        answer=candidate_answer if confidence != ConfidenceLevel.low else f"{candidate_answer}\n\n注意：当前视频证据一致性较低，建议二次检索或人工复核。",
        evidence_segment_ids=final_ids,
        timestamps=[format_timestamp(segment.start) for segment in final_segments],
        quotes=[segment.text[:240] for segment in final_segments],
        confidence=confidence,
        needs_visual_check=question_needs_visual_check(question),
        reason=f"query/evidence dual-DPP agreement={agreement:.2f}; low confidence is not proof of falsehood.",
    )


def _compose_candidate_answer(segments: list[TranscriptSegment]) -> str:
    claims = [" ".join(segment.text.split())[:220] for segment in segments[:3]]
    return "基于当前视频证据：" + " ".join(claims)
