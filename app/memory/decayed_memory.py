from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from app.config import Settings, get_settings
from app.retrieval.embeddings import HashingFallbackEmbedder, embed_texts

MemoryType = Literal["speech", "frame_caption", "user_preference", "model_summary", "user_verified", "derived_insight"]


@dataclass
class MemoryItem:
    memory_id: str
    content: str
    memory_type: MemoryType
    timestamp: str
    frequency: int = 1
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightedMemory:
    item: MemoryItem
    similarity: float
    time_weight: float
    frequency_weight: float
    importance_weight: float
    final_weight: float


class DecayedMemoryStore:
    """Vector memory with time decay and frequency-weighted retrieval.

    This module is intentionally separate from video evidence retrieval. It can
    feed extra context to QA/ConversationAgent while remaining compatible with
    Query-DPP because it returns stable ids, scores, and memory types.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str = "agent_decayed_memory",
        default_alpha: float = 0.015,
        beta: float = 0.25,
        allow_fallback: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.default_alpha = default_alpha
        self.beta = beta
        self.allow_fallback = allow_fallback
        self.type_alpha: dict[str, float] = {
            "speech": default_alpha,
            "frame_caption": default_alpha * 0.8,
            "user_preference": default_alpha * 0.25,
            "model_summary": default_alpha * 0.6,
            "user_verified": default_alpha * 0.15,
            "derived_insight": default_alpha * 0.5,
        }
        self._fallback_embedder = HashingFallbackEmbedder()
        self._fallback_items: dict[str, tuple[MemoryItem, list[float]]] = {}
        self._init_chroma()

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        memory_id: str | None = None,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> MemoryItem:
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        item = MemoryItem(
            memory_id=memory_id or f"mem_{uuid4().hex}",
            content=content.strip(),
            memory_type=memory_type,
            timestamp=(timestamp or datetime.now(UTC)).isoformat(),
            frequency=1,
            importance=max(0.0, float(importance)),
            metadata=metadata or {},
        )
        embedding = self._embed([item.content])[0]
        if self.using_chroma:
            self.collection.upsert(
                ids=[item.memory_id],
                documents=[item.content],
                embeddings=[embedding],
                metadatas=[self._metadata_for_item(item)],
            )
        else:
            self._fallback_items[item.memory_id] = (item, embedding)
        return item

    def search(
        self,
        query: str,
        top_k: int = 20,
        top_n: int = 8,
        memory_types: list[MemoryType] | None = None,
        update_frequency: bool = True,
    ) -> list[WeightedMemory]:
        if not query.strip():
            return []
        query_embedding = self._embed([query])[0]
        candidates = self._search_candidates(query_embedding, top_k=top_k, memory_types=memory_types)
        weighted = [self._weight_candidate(item, similarity) for item, similarity in candidates]
        selected = sorted(weighted, key=lambda row: row.final_weight, reverse=True)[:top_n]
        if update_frequency:
            self.mark_used([row.item.memory_id for row in selected])
        return selected

    def mark_used(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        if self.using_chroma:
            existing = self.collection.get(ids=memory_ids, include=["documents", "metadatas", "embeddings"])
            ids = existing.get("ids") or []
            docs = existing.get("documents") or []
            metas = existing.get("metadatas") or []
            embeddings = existing.get("embeddings")
            if embeddings is None:
                embeddings = self._embed([str(doc or "") for doc in docs])
            updated_metas = []
            for meta in metas:
                next_meta = dict(meta or {})
                next_meta["frequency"] = int(next_meta.get("frequency") or 1) + 1
                next_meta["last_accessed_at"] = datetime.now(UTC).isoformat()
                updated_metas.append(next_meta)
            if ids:
                self.collection.upsert(ids=ids, documents=docs, metadatas=updated_metas, embeddings=embeddings)
            return

        for memory_id in memory_ids:
            row = self._fallback_items.get(memory_id)
            if not row:
                continue
            item, embedding = row
            item.frequency += 1
            item.metadata["last_accessed_at"] = datetime.now(UTC).isoformat()
            self._fallback_items[memory_id] = (item, embedding)

    def cleanup(self, min_weight: float = 0.01, memory_types: list[MemoryType] | None = None) -> int:
        """Delete memories whose time/frequency/importance score is too low."""
        candidates = self._all_items(memory_types=memory_types)
        to_delete = []
        for item, _embedding in candidates:
            weighted = self._weight_candidate(item, similarity=1.0)
            if weighted.final_weight < min_weight:
                to_delete.append(item.memory_id)
        if not to_delete:
            return 0
        if self.using_chroma:
            self.collection.delete(ids=to_delete)
        else:
            for memory_id in to_delete:
                self._fallback_items.pop(memory_id, None)
        return len(to_delete)

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self.collection = client.get_or_create_collection(self.collection_name, metadata={"scope": "decayed_memory"})
            self.using_chroma = True
        except Exception as exc:  # noqa: BLE001
            if not self.allow_fallback:
                raise RuntimeError(f"memory_vector_store_unavailable: {exc}") from exc
            self.collection = None
            self.using_chroma = False

    def _search_candidates(
        self,
        query_embedding: list[float],
        top_k: int,
        memory_types: list[MemoryType] | None,
    ) -> list[tuple[MemoryItem, float]]:
        allowed_types = set(memory_types or [])
        if self.using_chroma:
            where = {"memory_type": {"$in": sorted(allowed_types)}} if allowed_types else None
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            docs = (result.get("documents") or [[]])[0]
            metas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            rows = []
            for memory_id, doc, meta, distance in zip(ids, docs, metas, distances, strict=False):
                item = self._item_from_chroma(memory_id, doc, meta)
                rows.append((item, max(0.0, 1.0 - float(distance or 0.0))))
            return rows

        scored: list[tuple[MemoryItem, float]] = []
        for item, embedding in self._fallback_items.values():
            if allowed_types and item.memory_type not in allowed_types:
                continue
            scored.append((item, _cosine(query_embedding, embedding)))
        return sorted(scored, key=lambda row: row[1], reverse=True)[:top_k]

    def _all_items(self, memory_types: list[MemoryType] | None = None) -> list[tuple[MemoryItem, list[float] | None]]:
        allowed_types = set(memory_types or [])
        if self.using_chroma:
            where = {"memory_type": {"$in": sorted(allowed_types)}} if allowed_types else None
            result = self.collection.get(where=where, include=["documents", "metadatas"])
            return [
                (self._item_from_chroma(memory_id, doc, meta), None)
                for memory_id, doc, meta in zip(result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or [], strict=False)
            ]
        return [(item, embedding) for item, embedding in self._fallback_items.values() if not allowed_types or item.memory_type in allowed_types]

    def _weight_candidate(self, item: MemoryItem, similarity: float) -> WeightedMemory:
        age_days = max(0.0, (datetime.now(UTC) - _parse_dt(item.timestamp)).total_seconds() / 86400.0)
        alpha = self.type_alpha.get(item.memory_type, self.default_alpha)
        time_weight = math.exp(-alpha * age_days)
        frequency_weight = 1.0 + self.beta * math.log(max(1, item.frequency) + 1)
        importance_weight = max(0.0, item.importance)
        final_weight = max(0.0, similarity) * time_weight * frequency_weight * importance_weight
        return WeightedMemory(
            item=item,
            similarity=similarity,
            time_weight=time_weight,
            frequency_weight=frequency_weight,
            importance_weight=importance_weight,
            final_weight=final_weight,
        )

    def _metadata_for_item(self, item: MemoryItem) -> dict[str, Any]:
        metadata = {
            "memory_id": item.memory_id,
            "memory_type": item.memory_type,
            "timestamp": item.timestamp,
            "frequency": item.frequency,
            "importance": item.importance,
        }
        for key, value in item.metadata.items():
            if isinstance(value, str | int | float | bool) or value is None:
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, ensure_ascii=False)
        return metadata

    def _item_from_chroma(self, memory_id: str, document: str, metadata: dict[str, Any]) -> MemoryItem:
        meta = dict(metadata or {})
        return MemoryItem(
            memory_id=str(meta.get("memory_id") or memory_id),
            content=str(document or ""),
            memory_type=str(meta.get("memory_type") or "model_summary"),  # type: ignore[arg-type]
            timestamp=str(meta.get("timestamp") or datetime.now(UTC).isoformat()),
            frequency=int(meta.get("frequency") or 1),
            importance=float(meta.get("importance") or 1.0),
            metadata={key: value for key, value in meta.items() if key not in {"memory_id", "memory_type", "timestamp", "frequency", "importance"}},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return embed_texts(texts, self.settings)
        except Exception:
            if not self.allow_fallback:
                raise
            return self._fallback_embedder.embed_texts(texts)


def weighted_memory_to_dict(row: WeightedMemory) -> dict[str, Any]:
    return {
        "memory_id": row.item.memory_id,
        "content": row.item.content,
        "memory_type": row.item.memory_type,
        "timestamp": row.item.timestamp,
        "frequency": row.item.frequency,
        "importance": row.item.importance,
        "weights": {
            "similarity": round(row.similarity, 6),
            "time": round(row.time_weight, 6),
            "frequency": round(row.frequency_weight, 6),
            "importance": round(row.importance_weight, 6),
            "final": round(row.final_weight, 6),
        },
        "metadata": row.item.metadata,
    }


def _parse_dt(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo decayed memory retrieval.")
    parser.add_argument("--query", default="agent memory retrieval", help="Query to search.")
    parser.add_argument("--add-demo", action="store_true", help="Insert demo memories before searching.")
    args = parser.parse_args()

    store = DecayedMemoryStore()
    if args.add_demo:
        store.add_memory("Agent memory should keep evidence grounded retrieval results.", "model_summary", importance=1.2)
        store.add_memory("User prefers concise answers with timestamp evidence.", "user_preference", importance=1.5)
        store.add_memory("A frame caption described a memory pipeline diagram with retrieval and reflection.", "frame_caption")
    rows = store.search(args.query)
    print(json.dumps([weighted_memory_to_dict(row) for row in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
