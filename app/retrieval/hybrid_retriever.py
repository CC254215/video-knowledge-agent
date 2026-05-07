from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.models import MultimodalSegment, RetrievedEvidence
from app.retrieval.bm25_store import BM25Store
from app.retrieval.chroma_memory import ChromaMemoryStore


@dataclass
class _RankedResult:
    evidence: RetrievedEvidence
    rank: int
    source: str


class HybridEvidenceRetriever:
    """Two-way retrieval over Chroma vectors and BM25 with reciprocal rank fusion."""

    def __init__(self, vector_store: ChromaMemoryStore, settings: Settings) -> None:
        self.vector_store = vector_store
        self.settings = settings
        self.last_trace: dict[str, object] = {}

    def search(
        self,
        video_id: str,
        query: str,
        segments: list[MultimodalSegment],
        top_k: int = 8,
        evidence_types: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        mode = self.settings.retrieval_mode.lower()
        if mode in {"vector", "chroma"}:
            results = self.vector_store.search(video_id, query, top_k=top_k, evidence_types=evidence_types)
            self.last_trace = _trace_payload(mode, query, results, [], results)
            return results
        if mode == "bm25":
            results = BM25Store(video_id, segments).search(query, top_k=top_k, evidence_types=evidence_types)
            self.last_trace = _trace_payload(mode, query, [], results, results)
            return results

        vector_results = self._vector_search(video_id, query, top_k, evidence_types)
        bm25_results = self._bm25_search(video_id, query, segments, top_k, evidence_types)
        fused = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k=top_k,
            rrf_k=self.settings.rrf_k,
            vector_weight=self.settings.vector_rrf_weight,
            bm25_weight=self.settings.bm25_rrf_weight,
        )
        self.last_trace = _trace_payload(mode, query, vector_results, bm25_results, fused)
        return fused

    def _vector_search(
        self,
        video_id: str,
        query: str,
        top_k: int,
        evidence_types: list[str] | None,
    ) -> list[RetrievedEvidence]:
        try:
            return self.vector_store.search(
                video_id,
                query,
                top_k=max(top_k, self.settings.vector_top_k),
                evidence_types=evidence_types,
            )
        except Exception:
            return []

    def _bm25_search(
        self,
        video_id: str,
        query: str,
        segments: list[MultimodalSegment],
        top_k: int,
        evidence_types: list[str] | None,
    ) -> list[RetrievedEvidence]:
        try:
            return BM25Store(video_id, segments).search(
                query,
                top_k=max(top_k, self.settings.bm25_top_k),
                evidence_types=evidence_types,
            )
        except Exception:
            return []


def reciprocal_rank_fusion(
    vector_results: list[RetrievedEvidence],
    bm25_results: list[RetrievedEvidence],
    top_k: int,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[RetrievedEvidence]:
    rows: dict[str, RetrievedEvidence] = {}
    scores: dict[str, float] = {}
    ranked_rows = [
        *[_RankedResult(item, rank, "vector") for rank, item in enumerate(vector_results, start=1)],
        *[_RankedResult(item, rank, "bm25") for rank, item in enumerate(bm25_results, start=1)],
    ]
    for ranked in ranked_rows:
        evidence_id = ranked.evidence.evidence_id
        rows.setdefault(evidence_id, ranked.evidence)
        weight = vector_weight if ranked.source == "vector" else bm25_weight
        scores[evidence_id] = scores.get(evidence_id, 0.0) + weight / (rrf_k + ranked.rank)

    if not scores:
        return []
    max_score = max(scores.values()) or 1.0
    ordered_ids = sorted(scores, key=lambda evidence_id: scores[evidence_id], reverse=True)[:top_k]
    fused: list[RetrievedEvidence] = []
    for evidence_id in ordered_ids:
        item = rows[evidence_id].model_copy()
        item.score = max(0.0, min(1.0, scores[evidence_id] / max_score))
        fused.append(item)
    return fused


def _trace_payload(
    mode: str,
    query: str,
    vector_results: list[RetrievedEvidence],
    bm25_results: list[RetrievedEvidence],
    fused_results: list[RetrievedEvidence],
) -> dict[str, object]:
    return {
        "mode": mode,
        "query": query,
        "vector_results": _trace_rows(vector_results),
        "bm25_results": _trace_rows(bm25_results),
        "fused_results": _trace_rows(fused_results),
    }


def _trace_rows(rows: list[RetrievedEvidence]) -> list[dict[str, object]]:
    return [
        {
            "evidence_id": item.evidence_id,
            "segment_id": item.segment_id,
            "evidence_type": item.evidence_type,
            "start": item.start,
            "end": item.end,
            "score": item.score,
        }
        for item in rows[:20]
    ]
