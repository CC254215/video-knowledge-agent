from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from app.models import MultimodalSegment, RetrievedEvidence


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class BM25Document:
    evidence_id: str
    document: str
    metadata: dict[str, Any]
    tokens: list[str]


class BM25Store:
    """Small per-video BM25 index built from loaded multimodal segments."""

    def __init__(self, video_id: str, segments: list[MultimodalSegment], k1: float = 1.5, b: float = 0.75) -> None:
        self.video_id = video_id
        self.k1 = k1
        self.b = b
        self.documents = _documents_for_segments(video_id, segments)
        self.doc_count = len(self.documents)
        self.avg_doc_len = (
            sum(len(document.tokens) for document in self.documents) / self.doc_count if self.doc_count else 0.0
        )
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []
        for document in self.documents:
            freqs: dict[str, int] = {}
            for token in document.tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.term_freqs.append(freqs)
            for token in freqs:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(
        self,
        query: str,
        top_k: int = 8,
        evidence_types: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        if not query.strip() or not self.documents:
            return []
        allowed_types = {item for item in (evidence_types or ["speech", "frame_caption"]) if item in {"speech", "frame_caption", "frame"}}
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored: list[tuple[int, float]] = []
        query_terms = sorted(set(query_tokens))
        for index, document in enumerate(self.documents):
            if allowed_types and document.metadata.get("evidence_type") not in allowed_types:
                continue
            score = self._score(index, query_terms)
            score += _exact_match_bonus(query, document.document, query_terms)
            if score > 0:
                scored.append((index, score))
        if not scored:
            return []
        max_score = max(score for _, score in scored) or 1.0
        rows = sorted(scored, key=lambda row: row[1], reverse=True)[:top_k]
        return [_result_from_document(self.video_id, self.documents[index], score / max_score) for index, score in rows]

    def _score(self, index: int, query_terms: list[str]) -> float:
        freqs = self.term_freqs[index]
        doc_len = len(self.documents[index].tokens)
        score = 0.0
        for token in query_terms:
            tf = freqs.get(token, 0)
            if not tf:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * (tf * (self.k1 + 1.0)) / denom
        return score


def tokenize(text: str) -> list[str]:
    base_tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    chinese_chars = [token for token in base_tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [left + right for left, right in zip(chinese_chars, chinese_chars[1:], strict=False)]
    return base_tokens + bigrams


def _documents_for_segments(video_id: str, segments: list[MultimodalSegment]) -> list[BM25Document]:
    rows: list[BM25Document] = []
    for segment in segments:
        if segment.transcript_text.strip():
            metadata = {
                "video_id": video_id,
                "segment_id": segment.segment_id,
                "start": segment.start,
                "end": segment.end,
                "evidence_type": "speech",
                "modality": "speech",
                "raw_text": segment.raw_transcript_text or segment.transcript_text,
                "normalized_text": segment.normalized_transcript_text or segment.transcript_text,
            }
            rows.append(
                BM25Document(
                    evidence_id=f"{video_id}:{segment.segment_id}:speech",
                    document=segment.transcript_text,
                    metadata=metadata,
                    tokens=tokenize(segment.transcript_text),
                )
            )
        for caption in segment.visual_captions:
            if not caption.caption.strip():
                continue
            metadata = {
                "video_id": video_id,
                "segment_id": segment.segment_id,
                "frame_id": caption.frame_id,
                "timestamp": caption.timestamp,
                "start": segment.start,
                "end": segment.end,
                "evidence_type": "frame_caption",
                "modality": "frame_caption",
            }
            if caption.image_path:
                metadata["image_path"] = caption.image_path
            rows.append(
                BM25Document(
                    evidence_id=f"{video_id}:{segment.segment_id}:frame_caption:{caption.frame_id}",
                    document=caption.caption,
                    metadata=metadata,
                    tokens=tokenize(caption.caption),
                )
            )
    return rows


def _exact_match_bonus(query: str, document: str, query_terms: list[str]) -> float:
    query_lower = query.lower()
    document_lower = document.lower()
    bonus = 0.0
    for term in query_terms:
        if len(term) >= 3 and term in document_lower:
            bonus += 0.08
    if query_lower.strip() and query_lower.strip() in document_lower:
        bonus += 0.2
    return bonus


def _result_from_document(video_id: str, document: BM25Document, score: float) -> RetrievedEvidence:
    metadata = document.metadata
    return RetrievedEvidence(
        evidence_id=document.evidence_id,
        video_id=video_id,
        segment_id=str(metadata.get("segment_id")),
        evidence_type=metadata.get("evidence_type"),
        text=document.document,
        start=float(metadata.get("start") or 0.0),
        end=float(metadata.get("end") or metadata.get("start") or 0.0),
        frame_id=metadata.get("frame_id"),
        image_path=metadata.get("image_path"),
        timestamp=float(metadata["timestamp"]) if metadata.get("timestamp") is not None else None,
        score=max(0.0, min(1.0, score)),
        raw_text=metadata.get("raw_text"),
        normalized_text=metadata.get("normalized_text") or document.document,
    )
