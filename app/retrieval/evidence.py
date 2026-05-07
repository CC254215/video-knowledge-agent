from __future__ import annotations

from app.models import EvidenceSet, MultimodalEvidenceSet, MultimodalSegment, RetrievedEvidence
from app.reasoning.summarizer import format_timestamp
from app.retrieval.embeddings import HashingFallbackEmbedder
from app.retrieval.vector_store import InMemoryVectorStore


def compute_evidence_agreement(query_ids: list[str], evidence_ids: list[str]) -> float:
    if not query_ids or not evidence_ids:
        return 0.0
    return len(set(query_ids) & set(evidence_ids)) / min(len(set(query_ids)), len(set(evidence_ids)))


def compute_multimodal_evidence_agreement(
    query_ids: list[str],
    evidence_ids: list[str],
    query_ranges: list[tuple[float, float]] | None = None,
    evidence_ranges: list[tuple[float, float]] | None = None,
    semantic_overlap: float = 0.0,
) -> float:
    segment_overlap = compute_evidence_agreement(query_ids, evidence_ids)
    time_overlap = _time_range_overlap_score(query_ranges or [], evidence_ranges or [])
    return max(0.0, min(1.0, 0.5 * segment_overlap + 0.3 * time_overlap + 0.20 * semantic_overlap))


def compute_retrieved_evidence_agreement(query_evidence: list[RetrievedEvidence], answer_evidence: list[RetrievedEvidence]) -> float:
    query_ids = [item.segment_id for item in query_evidence]
    answer_ids = [item.segment_id for item in answer_evidence]
    query_ranges = [(item.start, item.end) for item in query_evidence]
    answer_ranges = [(item.start, item.end) for item in answer_evidence]
    # TODO: replace score approximation with real semantic overlap between selected evidence sets.
    semantic_overlap = 0.0
    if query_evidence and answer_evidence:
        semantic_overlap = min(1.0, sum(item.score for item in answer_evidence) / len(answer_evidence))
    return compute_multimodal_evidence_agreement(query_ids, answer_ids, query_ranges, answer_ranges, semantic_overlap)


def confidence_from_agreement(agreement: float) -> str:
    if agreement >= 0.6:
        return "high"
    if agreement >= 0.3:
        return "medium"
    return "low"


class EvidenceRetriever:
    def __init__(self, store: InMemoryVectorStore) -> None:
        self.store = store

    def retrieve_query_evidence(self, video_id: str, question: str, top_k: int = 5) -> EvidenceSet:
        results = self.store.search(video_id, question, top_k=top_k)
        return EvidenceSet(
            video_id=video_id,
            query=question,
            segment_ids=[result.segment.segment_id for result in results],
            scores={result.segment.segment_id: result.score for result in results},
            quotes={result.segment.segment_id: result.segment.text for result in results},
        )

    def retrieve_answer_evidence(self, video_id: str, candidate_answer: str, top_k: int = 5) -> EvidenceSet:
        return self.retrieve_query_evidence(video_id, candidate_answer, top_k=top_k)


class MultimodalEvidenceRetriever:
    def __init__(self, video_id: str, segments: list[MultimodalSegment]) -> None:
        self.video_id = video_id
        self.segments = segments
        self.embedder = HashingFallbackEmbedder()

    def retrieve_query_evidence(self, query: str, top_k: int = 5) -> MultimodalEvidenceSet:
        scored = self._score(query)
        return self._to_evidence_set(query, scored[:top_k])

    def retrieve_answer_evidence(self, candidate_answer: str, top_k: int = 5) -> MultimodalEvidenceSet:
        scored = self._score(candidate_answer)
        return self._to_evidence_set(candidate_answer, scored[:top_k])

    def _score(self, query: str) -> list[tuple[MultimodalSegment, float]]:
        query_vector = self.embedder.embed_texts([query])[0]
        rows: list[tuple[MultimodalSegment, float]] = []
        for segment in self.segments:
            text = multimodal_segment_text(segment)
            segment_vector = self.embedder.embed_texts([text])[0]
            score = _cosine_lists(query_vector, segment_vector)
            if any(token and token.lower() in text.lower() for token in query.split()):
                score += 0.05
            rows.append((segment, score))
        return sorted(rows, key=lambda item: item[1], reverse=True)

    def _to_evidence_set(self, query: str, rows: list[tuple[MultimodalSegment, float]]) -> MultimodalEvidenceSet:
        evidence_types: list[str] = []
        frame_ids: list[str] = []
        timestamps: list[str] = []
        scores: dict[str, float] = {}
        quotes: dict[str, str] = {}
        ranges: dict[str, tuple[float, float]] = {}
        for segment, score in rows:
            for evidence_type in segment.evidence_types:
                if evidence_type not in evidence_types:
                    evidence_types.append(evidence_type)
            frame_ids.extend(frame_id for frame_id in segment.representative_frame_ids if frame_id not in frame_ids)
            timestamps.append(format_timestamp(segment.start))
            scores[segment.segment_id] = score
            quotes[segment.segment_id] = multimodal_segment_text(segment)[:500]
            ranges[segment.segment_id] = (segment.start, segment.end)
        return MultimodalEvidenceSet(
            video_id=self.video_id,
            query=query,
            segment_ids=[segment.segment_id for segment, _ in rows],
            frame_ids=frame_ids,
            evidence_types=evidence_types,  # type: ignore[arg-type]
            timestamps=timestamps,
            scores=scores,
            quotes=quotes,
            time_ranges=ranges,
        )


def multimodal_segment_text(segment: MultimodalSegment) -> str:
    captions = " ".join(caption.caption for caption in segment.visual_captions if caption.caption)
    return "\n".join(part for part in [segment.transcript_text, segment.visual_summary, captions] if part.strip())


def _cosine_lists(left: list[float], right: list[float]) -> float:
    import math

    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _time_range_overlap_score(left: list[tuple[float, float]], right: list[tuple[float, float]]) -> float:
    if not left or not right:
        return 0.0
    overlaps = 0
    for start_a, end_a in left:
        for start_b, end_b in right:
            if max(start_a, start_b) <= min(end_a, end_b):
                overlaps += 1
                break
    return overlaps / min(len(left), len(right))
