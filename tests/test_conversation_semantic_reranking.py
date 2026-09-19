from app.config import Settings
from app.models import RetrievedEvidence
from app.reasoning.conversation_agent import ConversationAgent
from app.retrieval import embeddings


def _candidates():
    return [
        RetrievedEvidence(
            evidence_id=f"v:seg_{index}:speech", video_id="v",
            segment_id=f"seg_{index}", evidence_type="speech", text=text,
            start=float(index), end=float(index + 1), score=score,
        )
        for index, (text, score) in enumerate([
            ("降低训练成本", 1.0), ("减少训练花费", 0.99), ("保存检查点以恢复任务", 0.9),
        ])
    ]


def test_semantic_reranking_reuses_configured_embedding_cache(tmp_path, monkeypatch):
    calls = []

    def remote(self, texts):
        calls.append((self.settings.embedding_model, texts))
        # Distinct wording, same meaning: first two candidates are redundant.
        vectors = {"降低训练成本": [1.0, 0.0], "减少训练花费": [1.0, 0.0], "保存检查点以恢复任务": [0.0, 1.0]}
        return [vectors[text] for text in texts]

    monkeypatch.setattr(embeddings.OpenAICompatibleEmbedder, "embed_texts", remote)
    settings = Settings(_env_file=None, openai_api_key="test-key", embedding_model="Embedding-3",
                        embedding_cache_enabled=True, force_refresh=False,
                        embedding_cache_path=tmp_path / "cache.sqlite3")
    agent = ConversationAgent("v", tmp_path, settings=settings, store=object())
    for _ in range(2):
        selected = agent._select_evidence(_candidates(), k=2)
        assert [item.segment_id for item in selected] == ["seg_0", "seg_2"]
    assert calls == [("Embedding-3", [item.text for item in _candidates()])]


def test_semantic_reranking_degrades_on_embedding_outage(tmp_path, monkeypatch):
    def unavailable(self, texts):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(embeddings.OpenAICompatibleEmbedder, "embed_texts", unavailable)
    settings = Settings(_env_file=None, openai_api_key="test-key", embedding_model="Embedding-3",
                        embedding_cache_path=tmp_path / "cache.sqlite3")
    agent = ConversationAgent("v", tmp_path, settings=settings, store=object())
    assert agent._select_evidence([], k=2) == []
    selected = agent._select_evidence(_candidates(), k=2)
    assert len(selected) == 2
    assert selected[0].segment_id == "seg_0"
    assert all(item.evidence_id in {c.evidence_id for c in _candidates()} for item in selected)
