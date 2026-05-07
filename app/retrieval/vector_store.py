from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.models import TranscriptSegment
from app.retrieval.embeddings import HashingFallbackEmbedder


@dataclass
class SearchResult:
    segment: TranscriptSegment
    score: float
    embedding: list[float]


@dataclass
class InMemoryVectorStore:
    embedder: object = field(default_factory=HashingFallbackEmbedder)
    segments: dict[str, dict[str, TranscriptSegment]] = field(default_factory=dict)
    embeddings: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def add_segments(self, video_id: str, segments: list[TranscriptSegment]) -> None:
        texts = [segment.text for segment in segments]
        vectors = self.embedder.embed_texts(texts)  # type: ignore[attr-defined]
        self.segments.setdefault(video_id, {})
        self.embeddings.setdefault(video_id, {})
        for segment, vector in zip(segments, vectors, strict=True):
            self.segments[video_id][segment.segment_id] = segment
            self.embeddings[video_id][segment.segment_id] = vector

    def search(self, video_id: str, query: str, top_k: int = 5) -> list[SearchResult]:
        if video_id not in self.segments:
            return []
        query_vector = np.array(self.embedder.embed_texts([query])[0], dtype=float)  # type: ignore[attr-defined]
        results: list[SearchResult] = []
        for segment_id, segment in self.segments[video_id].items():
            vector = np.array(self.embeddings[video_id][segment_id], dtype=float)
            denom = np.linalg.norm(query_vector) * np.linalg.norm(vector)
            score = float(np.dot(query_vector, vector) / denom) if denom else 0.0
            results.append(SearchResult(segment=segment, score=score, embedding=vector.tolist()))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def get_segments(self, video_id: str, ids: list[str]) -> list[TranscriptSegment]:
        return [self.segments.get(video_id, {}).get(segment_id) for segment_id in ids if self.segments.get(video_id, {}).get(segment_id)]
