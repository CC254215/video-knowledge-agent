from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from typing import Protocol

import numpy as np

from app.config import Settings, get_settings
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)
EMBEDDING_CACHE_SCHEMA_VERSION = 1


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.settings.has_embedding_config:
            raise RuntimeError("Missing embedding config; set OPENAI_API_KEY, EMBEDDING_BASE_URL, and EMBEDDING_MODEL.")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for embedding API calls. Install project dependencies.") from exc
        url = _join_openai_endpoint(self.settings.runtime_embedding_base_url, "embeddings")

        def _request() -> httpx.Response:
            result = httpx.post(
                url,
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                json={"model": self.settings.embedding_model, "input": texts},
                timeout=self.settings.request_timeout_seconds,
            )
            if result.status_code == 429 or result.status_code >= 500:
                raise RuntimeError(f"retryable_http_status={result.status_code}, body={result.text[:1000]}")
            return result

        response = get_scheduler(self.settings).call("embedding", _request)
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]


class HashingFallbackEmbedder:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=float)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dim
                vec[index] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append((vec / norm if norm else vec).tolist())
        return vectors


def tfidf_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:  # pragma: no cover - sklearn/pandas wheels may be unavailable or ABI-incompatible
        logger.warning("TF-IDF fallback unavailable; using hashing embeddings: %s", exc)
        return HashingFallbackEmbedder().embed_texts(texts)
    matrix = TfidfVectorizer().fit_transform(texts)
    return matrix.toarray().tolist()


def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    settings = settings or get_settings()
    if settings.has_embedding_config:
        try:
            return _embed_texts_with_cache(texts, settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding API failed; falling back to local hashing embeddings: %s", exc)
    return HashingFallbackEmbedder().embed_texts(texts)


def _embed_texts_with_cache(texts: list[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []
    if not settings.embedding_cache_enabled or settings.force_refresh:
        return OpenAICompatibleEmbedder(settings).embed_texts(texts)

    cache = EmbeddingCache(settings)
    cached = cache.get_many(texts)
    output: list[list[float] | None] = [cached.get(_cache_key(settings, text)) for text in texts]
    missing_by_key: dict[str, str] = {}
    for text, vector in zip(texts, output, strict=False):
        if vector is None:
            missing_by_key.setdefault(_cache_key(settings, text), text)

    if missing_by_key:
        missing_texts = list(missing_by_key.values())
        fresh_vectors = OpenAICompatibleEmbedder(settings).embed_texts(missing_texts)
        fresh_by_key = {
            _cache_key(settings, text): vector
            for text, vector in zip(missing_texts, fresh_vectors, strict=False)
        }
        cache.set_many(fresh_by_key)
        for index, text in enumerate(texts):
            if output[index] is None:
                output[index] = fresh_by_key[_cache_key(settings, text)]

    return [vector for vector in output if vector is not None]


class EmbeddingCache:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.embedding_cache_path or (settings.data_dir / "cache" / "embeddings.sqlite3")

    def get_many(self, texts: list[str]) -> dict[str, list[float]]:
        keys = list({_cache_key(self.settings, text) for text in texts})
        return self._get_keys(keys)

    def _get_keys(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                rows = []
                for chunk in _chunks(keys, 500):
                    rows.extend(
                        conn.execute(
                            f"SELECT cache_key, vector_json FROM embeddings WHERE cache_key IN ({','.join('?' for _ in chunk)})",
                            chunk,
                        ).fetchall()
                    )
            output: dict[str, list[float]] = {}
            for cache_key, vector_json in rows:
                try:
                    vector = json.loads(str(vector_json))
                except json.JSONDecodeError:
                    continue
                if isinstance(vector, list):
                    output[str(cache_key)] = [float(item) for item in vector]
            return output
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding cache read failed; continuing without cache: %s", exc)
            return {}

    def set_many(self, vectors_by_key: dict[str, list[float]]) -> None:
        if not vectors_by_key:
            return
        now = time.time()
        rows = [
            (cache_key, json.dumps(vector), len(vector), now, now)
            for cache_key, vector in vectors_by_key.items()
        ]
        try:
            with self._connect() as conn:
                self._ensure_schema(conn)
                conn.executemany(
                    """
                    INSERT INTO embeddings(cache_key, vector_json, dimension, created_at, last_accessed_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        vector_json=excluded.vector_json,
                        dimension=excluded.dimension,
                        last_accessed_at=excluded.last_accessed_at
                    """,
                    rows,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding cache write failed; continuing without cache persistence: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                vector_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(EMBEDDING_CACHE_SCHEMA_VERSION),),
        )


def _cache_key(settings: Settings, text: str) -> str:
    payload = {
        "schema": EMBEDDING_CACHE_SCHEMA_VERSION,
        "provider": "openai-compatible",
        "base_url": settings.runtime_embedding_base_url,
        "model": settings.embedding_model,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"
