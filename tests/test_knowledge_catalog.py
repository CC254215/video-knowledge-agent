from __future__ import annotations

import json
import pytest

from app.config import Settings
from app.memory.knowledge_catalog import KnowledgeCatalog, historical_context
from app.memory.mempalace_adapter import MemPalaceMCPAdapter
from app.models import MultimodalSegment, Storyline, VideoMetadata
from app.reasoning.conversation_agent import validate_supplements


class Adapter:
    def __init__(self):
        self.rows = []

    def save_raw_evidence(self, payload):
        self.rows.append({"source_file": payload["source_file"], "similarity": 0.9})

    save_model_summary = save_raw_evidence

    def search_related_memories(self, query, limit=5):
        self.query = query
        return self.rows[:limit]


@pytest.fixture
def setup(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, mempalace_provider="mcp_stdio", mempalace_auto_status=False)
    return settings, KnowledgeCatalog(settings), Adapter()


def ingest(settings, catalog, video_id, text="Retrieval improves coverage but excessive candidates introduce noise.", count=1):
    segments = [MultimodalSegment(segment_id=f"seg_{i}", start=i*10, end=i*10+10,
                transcript_text=f"{text} Section {i}.", keywords=["retrieval"]) for i in range(count)]
    directory = settings.videos_dir / video_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "multimodal_segments.json").write_text(json.dumps([s.model_dump(mode="json") for s in segments]), encoding="utf-8")
    metadata = VideoMetadata(video_id=video_id, title=f"Retrieval {video_id}")
    return catalog.ingest_video(metadata, Storyline(video_id=video_id, query="", nodes=[]), segments)


def test_full_video_idempotence_and_source_changes(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old", count=25)
    assert catalog.stats()["active"] == 25
    assert catalog.sync(adapter)["saved"] == 25
    ingest(settings, catalog, "old", count=25)
    assert catalog.sync(adapter)["saved"] == 0
    old = catalog.resolve(adapter.rows[0])
    assert catalog.verify(old)
    ingest(settings, catalog, "old", text="Changed source")
    assert not catalog.resolve(adapter.rows[0])
    assert not catalog.verify(old)
    assert catalog.stats()["active"] == 1


def test_failed_write_stays_pending_and_retries(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old")
    class Broken(Adapter):
        def save_raw_evidence(self, payload):
            raise RuntimeError("offline")
    stats = catalog.sync(Broken())
    assert stats["pending"] == 1 and stats["saved"] == 0
    assert catalog.sync(adapter)["pending"] == 0


def test_cross_video_context_excludes_current_and_checks_sources(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old")
    ingest(settings, catalog, "current")
    catalog.sync(adapter)
    context = historical_context("What are limitations?", "current", "Retrieval", settings, adapter=adapter)
    assert context["status"] == "ready"
    assert len(context["long_term_memory"]) == 1
    assert context["long_term_memory"][0]["video_id"] == "old"
    assert "limitations" in adapter.query
    (settings.videos_dir / "old" / "multimodal_segments.json").unlink()
    assert not historical_context("question", "current", "Retrieval", settings, adapter=adapter)["long_term_memory"]


def test_unrelated_memory_and_budget_not_forced(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old")
    catalog.sync(adapter)
    adapter.rows[0]["similarity"] = 0.01
    assert not historical_context("unrelated", "new", "Different", settings, adapter=adapter)["long_term_memory"]
    adapter.rows[0]["similarity"] = 0.99
    assert not historical_context("retrieval", "new", "Retrieval", settings, current_chars=10, adapter=adapter)["long_term_memory"]


def test_opposing_claims_are_preserved(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "a", text="Increasing retrieval count improves answer quality.")
    ingest(settings, catalog, "b", text="Increasing retrieval count reduces answer quality.")
    catalog.sync(adapter)
    context = historical_context("retrieval count", "current", "Retrieval", settings, adapter=adapter)
    assert len(context["long_term_memory"]) == 2


def test_manual_topics_survive_reprocessing(setup):
    settings, catalog, _ = setup
    ingest(settings, catalog, "v")
    chosen = catalog.assign_topics("v", ["Computer science / Retrieval", "Evaluation"], manual=True)
    ingest(settings, catalog, "v")
    assert catalog.topics("v") == chosen


def test_outage_does_not_raise(setup):
    settings, _, _ = setup
    class Broken(Adapter):
        def search_related_memories(self, *args, **kwargs):
            raise RuntimeError("offline")
    result = historical_context("q", "v", "Title", settings, adapter=Broken())
    assert result["status"] == "unavailable" and not result["long_term_memory"]


def test_citations_require_known_id_and_verbatim_quote(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old")
    catalog.sync(adapter)
    context = historical_context("retrieval", "v", "Retrieval", settings, adapter=adapter)
    row = context["long_term_memory"][0]
    valid = {"citation_id": row["citation_id"], "quote": row["evidence"][0]["quote"], "text": "Historical limitation"}
    history, _ = validate_supplements({"historical_supplements": [valid]}, context)
    assert len(history) == 1
    for broken in [{**valid, "citation_id": "H:fake"}, {**valid, "quote": "Fabricated quote"}]:
        assert not validate_supplements({"historical_supplements": [broken]}, context)[0]


def test_adapter_preserves_payload_and_uses_real_schema(setup, monkeypatch):
    settings, _, _ = setup
    adapter = MemPalaceMCPAdapter(settings)
    calls = []
    monkeypatch.setattr(adapter, "call_tool", lambda name, args: calls.append((name, args)) or {"success": True})
    adapter.save_derived_insight({"video_id": "v", "answer": "insight", "evidence": [{"segment_id": "s"}]})
    args = calls[0][1]
    assert args["wing"] and args["room"] and args["source_file"]
    assert '"segment_id": "s"' in args["content"]


def test_mcp_write_error_is_not_reported_as_saved(setup, monkeypatch):
    settings, _, _ = setup
    adapter = MemPalaceMCPAdapter(settings)
    monkeypatch.setattr(adapter, "call_tool", lambda *a: {"error": "disk full"})
    with pytest.raises(Exception, match="disk full"):
        adapter.save_model_summary({"video_id": "v", "content": "test"})


def test_server_resolves_index_to_original_quote(setup):
    settings, catalog, adapter = setup
    ingest(settings, catalog, "old")
    catalog.sync(adapter)
    context = historical_context("retrieval", "v", "Title", settings, adapter=adapter)
    source = context["long_term_memory"][0]
    item = {"citation_id": source["citation_id"], "evidence_index": 0, "text": "Historical explanation"}
    valid, _ = validate_supplements({"historical_supplements": [item]}, context)
    assert valid[0]["quote"] == source["evidence"][0]["quote"]
    assert not validate_supplements({"historical_supplements": [{**item, "evidence_index": -1}]}, context)[0]


def test_classification_model_failure_falls_back_and_manual_wins(setup, monkeypatch):
    settings, catalog, _ = setup
    settings.openai_api_key = "test-key"
    monkeypatch.setattr("app.reasoning.llm_client.LLMClient.generate_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    ingest(settings, catalog, "v")
    assert catalog.topics("v")[0]["label"] == "retrieval"
    chosen = catalog.assign_topics("v", ["Science/Retrieval"], manual=True)
    ingest(settings, catalog, "v")
    assert catalog.topics("v") == chosen


def test_classification_cache_avoids_repeat_model_calls(setup, monkeypatch):
    settings, catalog, _ = setup
    settings.openai_api_key = "test-key"
    calls = []
    monkeypatch.setattr("app.reasoning.llm_client.LLMClient.generate_json", lambda *a, **k: calls.append(1) or {"topics": ["Computer science/Retrieval"]})
    ingest(settings, catalog, "v")
    ingest(settings, catalog, "v")
    assert len(calls) == 1
    assert catalog.topics("v")[0]["label"] == "computer science/retrieval"


def test_shared_secondary_topic_routes_to_other_video_room(setup):
    settings, catalog, _ = setup
    ingest(settings, catalog, "current")
    ingest(settings, catalog, "old")
    current = catalog.assign_topics("current", ["Games", "Games/Builds"], manual=True)
    other = catalog.assign_topics("old", ["Games/Builds"], manual=True)
    assert catalog.related_rooms("current", current) == [other[0]["id"]]


def test_legacy_aggregation_survives_memory_outage(setup, monkeypatch):
    from app.analysis.multi_video_aggregator import MultiVideoAggregator
    settings, _, _ = setup
    monkeypatch.setattr("app.analysis.multi_video_aggregator.create_memory_adapter", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
    assert MultiVideoAggregator(settings).save_to_memory([]) == 0
