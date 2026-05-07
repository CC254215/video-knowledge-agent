from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.retrieval import embeddings


def test_embed_texts_reuses_sqlite_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_embed(self, texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    monkeypatch.setattr(embeddings.OpenAICompatibleEmbedder, "embed_texts", fake_embed)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        embedding_model="embedding-test",
        data_dir=tmp_path,
        embedding_cache_path=tmp_path / "embeddings.sqlite3",
    )

    first = embeddings.embed_texts(["alpha", "beta", "alpha"], settings)
    second = embeddings.embed_texts(["alpha", "beta"], settings)

    assert first == [[5.0, 0.0], [4.0, 1.0], [5.0, 0.0]]
    assert second == [[5.0, 0.0], [4.0, 1.0]]
    assert calls == [["alpha", "beta"]]
    assert (tmp_path / "embeddings.sqlite3").exists()


def test_embed_texts_can_disable_cache(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    def fake_embed(self, texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [[1.0] for _ in texts]

    monkeypatch.setattr(embeddings.OpenAICompatibleEmbedder, "embed_texts", fake_embed)
    settings = Settings(
        _env_file=None,
        openai_api_key="test-key",
        embedding_model="embedding-test",
        data_dir=tmp_path,
        embedding_cache_enabled=False,
    )

    assert embeddings.embed_texts(["alpha"], settings) == [[1.0]]
    assert embeddings.embed_texts(["alpha"], settings) == [[1.0]]
    assert calls == 2
