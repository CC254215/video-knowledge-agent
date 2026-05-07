from __future__ import annotations

from app.memory.mempalace_adapter import build_memory_context


class FakeAdapter:
    def search_related_memories(self, query: str, limit: int = 5):
        return [
            {"content": "User prefers concise answers.", "memory_type": "user_preference"},
            {"content": "User prefers concise answers.", "memory_type": "user_preference"},
            {"content": "A prior video discussed agent memory retrieval.", "memory_type": "model_summary"},
        ][:limit]


def test_build_memory_context_deduplicates_and_labels_memory() -> None:
    context = build_memory_context("agent memory", adapter=FakeAdapter(), limit=3)

    rows = context["long_term_memory"]
    assert len(rows) == 2
    assert rows[0]["source"] == "mempalace"
    assert rows[0]["usage_rule"].startswith("May guide background context")
    assert context["answer_policy"]["video_claims_require_video_evidence"] is True
