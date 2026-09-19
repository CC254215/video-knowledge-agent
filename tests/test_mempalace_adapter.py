from __future__ import annotations

from app.memory.mempalace_adapter import MemPalaceMCPAdapter, build_memory_context


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


def test_native_adapter_files_verbatim_conversation_in_a_scoped_drawer() -> None:
    adapter = object.__new__(MemPalaceMCPAdapter)
    calls = []
    adapter.call_tool = lambda name, arguments: calls.append((name, arguments)) or {"success": True}  # type: ignore[method-assign]

    adapter.save_conversation_turn({
        "content": "User:\\nExplain HNSW.\\n\\nAssistant:\\nIt is an ANN index.",
        "room": "conversation:video-1",
        "source_file": "conversation/session-1/turn-1.jsonl",
    })

    assert calls == [("mempalace_add_drawer", {
        "content": "User:\\nExplain HNSW.\\n\\nAssistant:\\nIt is an ANN index.",
        "wing": "video_knowledge_agent",
        "room": "conversation:video-1",
        "source_file": "conversation/session-1/turn-1.jsonl",
        "added_by": "video_knowledge_agent",
    })]
