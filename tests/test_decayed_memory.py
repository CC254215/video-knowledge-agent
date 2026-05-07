from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.memory import decayed_memory
from app.memory.decayed_memory import DecayedMemoryStore
from app.retrieval.embeddings import HashingFallbackEmbedder


def test_decayed_memory_search_updates_frequency(tmp_path: Path, monkeypatch) -> None:
    embedder = HashingFallbackEmbedder()
    monkeypatch.setattr(decayed_memory, "embed_texts", lambda texts, settings=None: embedder.embed_texts(texts))
    settings = Settings(data_dir=tmp_path / "data", chroma_path=tmp_path / "chroma")
    store = DecayedMemoryStore(settings=settings, collection_name="test_decayed_memory")
    store.add_memory(
        "Agent memory retrieval should preserve evidence and timestamps.",
        "model_summary",
        memory_id="m_recent",
        importance=1.0,
    )

    result = store.search("memory retrieval evidence", top_k=5, top_n=1)

    assert result
    assert result[0].item.memory_id == "m_recent"
    assert result[0].final_weight > 0
    second = store.search("memory retrieval evidence", top_k=5, top_n=1)
    assert second[0].item.frequency >= 2


def test_decayed_memory_time_decay_affects_ranking(tmp_path: Path, monkeypatch) -> None:
    embedder = HashingFallbackEmbedder()
    monkeypatch.setattr(decayed_memory, "embed_texts", lambda texts, settings=None: embedder.embed_texts(texts))
    settings = Settings(data_dir=tmp_path / "data", chroma_path=tmp_path / "chroma")
    store = DecayedMemoryStore(settings=settings, collection_name="test_decayed_memory_decay", default_alpha=1.0)
    store.add_memory(
        "retrieval evidence memory",
        "speech",
        memory_id="old",
        timestamp=datetime.now(UTC) - timedelta(days=30),
    )
    store.add_memory(
        "retrieval evidence memory",
        "speech",
        memory_id="new",
        timestamp=datetime.now(UTC),
    )

    result = store.search("retrieval evidence memory", top_k=5, top_n=2, update_frequency=False)

    assert [row.item.memory_id for row in result][:2] == ["new", "old"]
