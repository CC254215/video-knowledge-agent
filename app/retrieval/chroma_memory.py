from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.models import MultimodalSegment, RetrievedEvidence
from app.retrieval.embeddings import HashingFallbackEmbedder, embed_texts


class ChromaMemoryError(RuntimeError):
    pass


@dataclass
class _FallbackCollection:
    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict[str, Any]] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)


class ChromaMemoryStore:
    def __init__(self, settings: Settings | None = None, allow_fallback: bool = True) -> None:
        self.settings = settings or get_settings()
        self.allow_fallback = allow_fallback
        self._collections: dict[str, Any] = {}
        self._fallback: dict[str, _FallbackCollection] = {}
        try:
            import chromadb  # type: ignore

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self.using_chroma = True
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise ChromaMemoryError(f"embedding_unavailable: chromadb is not installed or cannot start: {exc}") from exc
            self.client = None
            self.using_chroma = False
            self._fallback_embedder = HashingFallbackEmbedder()

    def create_video_collection(self, video_id: str):
        name = _collection_name(video_id)
        if self.using_chroma:
            collection = self.client.get_or_create_collection(name=name, metadata={"video_id": video_id})  # type: ignore[union-attr]
            self._collections[video_id] = collection
            return collection
        self._fallback.setdefault(video_id, _FallbackCollection())
        return self._fallback[video_id]

    def add_multimodal_segments(self, video_id: str, segments: list[MultimodalSegment]) -> None:
        rows = _documents_for_segments(video_id, segments)
        if not rows:
            self.create_video_collection(video_id)
            return
        if self.using_chroma:
            collection = self.create_video_collection(video_id)
            all_rows = rows
            changed_rows = self._changed_rows(collection, rows)
            if not changed_rows:
                return
            ids = [row["id"] for row in changed_rows]
            documents = [row["document"] for row in changed_rows]
            metadatas = [row["metadata"] for row in changed_rows]
            embeddings = self._embed(documents)
            try:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            except Exception as exc:  # noqa: BLE001
                if _is_dimension_mismatch(exc):
                    # Embedding dimension changes require rebuilding the
                    # collection once. This preserves correctness while the
                    # normal path remains incremental.
                    self.delete_video_collection(video_id)
                    collection = self.create_video_collection(video_id)
                    ids = [row["id"] for row in all_rows]
                    documents = [row["document"] for row in all_rows]
                    metadatas = [row["metadata"] for row in all_rows]
                    embeddings = HashingFallbackEmbedder().embed_texts(documents)
                    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                    return
                raise ChromaMemoryError(f"embedding_unavailable: failed to add documents to Chroma: {exc}") from exc
            return
        collection = self.create_video_collection(video_id)
        existing = {evidence_id: index for index, evidence_id in enumerate(collection.ids)}
        new_documents = [row["document"] for row in rows]
        new_embeddings = self._embed(new_documents)
        for row, embedding in zip(rows, new_embeddings, strict=False):
            index = existing.get(row["id"])
            if index is None:
                collection.ids.append(row["id"])
                collection.documents.append(row["document"])
                collection.metadatas.append(row["metadata"])
                collection.embeddings.append(embedding)
            elif collection.documents[index] != row["document"] or collection.metadatas[index] != row["metadata"]:
                collection.documents[index] = row["document"]
                collection.metadatas[index] = row["metadata"]
                collection.embeddings[index] = embedding

    def has_video_documents(self, video_id: str) -> bool:
        if self.using_chroma:
            try:
                return int(self.create_video_collection(video_id).count()) > 0
            except Exception:
                return False
        return bool(self._fallback.get(video_id) and self._fallback[video_id].ids)

    def search(
        self,
        video_id: str,
        query: str,
        top_k: int = 8,
        evidence_types: list[str] | None = None,
    ) -> list[RetrievedEvidence]:
        allowed_types = {item for item in (evidence_types or ["speech", "frame_caption"]) if item in {"speech", "frame_caption", "frame"}}
        query_embedding = self._embed([query])[0]
        if self.using_chroma:
            collection = self.create_video_collection(video_id)
            where = {"$and": [{"video_id": video_id}, {"evidence_type": {"$in": sorted(allowed_types)}}]} if allowed_types else {"video_id": video_id}
            try:
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(1, top_k),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:  # noqa: BLE001
                if _is_dimension_mismatch(exc):
                    self.delete_video_collection(video_id)
                    return []
                raise ChromaMemoryError(f"embedding_unavailable: Chroma search failed: {exc}") from exc
            return _results_from_chroma(video_id, result)

        collection = self._fallback.get(video_id)
        if not collection:
            return []
        scored: list[tuple[int, float]] = []
        for index, metadata in enumerate(collection.metadatas):
            if metadata.get("evidence_type") not in allowed_types:
                continue
            score = _cosine(query_embedding, collection.embeddings[index])
            scored.append((index, score))
        rows = sorted(scored, key=lambda row: row[1], reverse=True)[:top_k]
        return [_fallback_result(video_id, collection, index, score) for index, score in rows]

    def delete_video_collection(self, video_id: str) -> None:
        if self.using_chroma:
            try:
                self.client.delete_collection(_collection_name(video_id))  # type: ignore[union-attr]
            except Exception:
                pass
            self._collections.pop(video_id, None)
        self._fallback.pop(video_id, None)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return embed_texts(texts, self.settings)
        except Exception as exc:  # noqa: BLE001
            if self.allow_fallback:
                return self._fallback_embedder.embed_texts(texts) if hasattr(self, "_fallback_embedder") else HashingFallbackEmbedder().embed_texts(texts)
            raise ChromaMemoryError(f"embedding_unavailable: {exc}") from exc

    def _changed_rows(self, collection: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            existing = collection.get(ids=[row["id"] for row in rows], include=["documents", "metadatas"])
        except Exception:
            return rows
        existing_docs = dict(zip(existing.get("ids") or [], existing.get("documents") or [], strict=False))
        existing_meta = dict(zip(existing.get("ids") or [], existing.get("metadatas") or [], strict=False))
        return [
            row
            for row in rows
            if existing_docs.get(row["id"]) != row["document"] or existing_meta.get(row["id"]) != row["metadata"]
        ]


def _documents_for_segments(video_id: str, segments: list[MultimodalSegment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if segment.transcript_text.strip():
            rows.append(
                {
                    "id": f"{video_id}:{segment.segment_id}:speech",
                    "document": segment.transcript_text,
                    "metadata": {
                        "video_id": video_id,
                        "segment_id": segment.segment_id,
                        "start": segment.start,
                        "end": segment.end,
                        "evidence_type": "speech",
                        "modality": "speech",
                    },
                }
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
                {
                    "id": f"{video_id}:{segment.segment_id}:frame_caption:{caption.frame_id}",
                    "document": caption.caption,
                    "metadata": metadata,
                }
            )
    return rows


def _results_from_chroma(video_id: str, result: dict[str, Any]) -> list[RetrievedEvidence]:
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    rows: list[RetrievedEvidence] = []
    for evidence_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
        rows.append(
            RetrievedEvidence(
                evidence_id=str(evidence_id),
                video_id=video_id,
                segment_id=str(metadata.get("segment_id")),
                evidence_type=metadata.get("evidence_type"),
                text=str(document or ""),
                start=float(metadata.get("start") or 0.0),
                end=float(metadata.get("end") or metadata.get("start") or 0.0),
                frame_id=metadata.get("frame_id"),
                image_path=metadata.get("image_path"),
                timestamp=float(metadata["timestamp"]) if metadata.get("timestamp") is not None else None,
                score=max(0.0, 1.0 - float(distance or 0.0)),
            )
        )
    return rows


def _fallback_result(video_id: str, collection: _FallbackCollection, index: int, score: float) -> RetrievedEvidence:
    metadata = collection.metadatas[index]
    return RetrievedEvidence(
        evidence_id=collection.ids[index],
        video_id=video_id,
        segment_id=str(metadata.get("segment_id")),
        evidence_type=metadata.get("evidence_type"),
        text=collection.documents[index],
        start=float(metadata.get("start") or 0.0),
        end=float(metadata.get("end") or metadata.get("start") or 0.0),
        frame_id=metadata.get("frame_id"),
        image_path=metadata.get("image_path"),
        timestamp=float(metadata["timestamp"]) if metadata.get("timestamp") is not None else None,
        score=score,
    )


def _collection_name(video_id: str, prefix: str = "") -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in video_id)
    name = f"video_{safe}" if not prefix else f"video_{prefix}_{safe}"
    return name[:63]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _is_dimension_mismatch(exc: Exception) -> bool:
    text = str(exc).lower()
    return "expecting embedding with dimension" in text or "dimension" in text and "embedding" in text
