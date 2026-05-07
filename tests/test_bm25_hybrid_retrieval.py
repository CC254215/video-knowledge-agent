from __future__ import annotations

from app.config import Settings
from app.models import FrameCaption, MultimodalSegment, RetrievedEvidence
from app.reasoning.conversation_agent import ConversationAgent
from app.retrieval.bm25_store import BM25Store, tokenize
from app.retrieval.hybrid_retriever import HybridEvidenceRetriever, reciprocal_rank_fusion


def test_bm25_store_matches_exact_config_token() -> None:
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=10,
            transcript_text="The runtime setting API_BASE_URL controls the endpoint.",
        ),
        MultimodalSegment(
            segment_id="seg_1",
            start=10,
            end=20,
            transcript_text="This part only discusses general video retrieval.",
        ),
    ]

    results = BM25Store("v1", segments).search("API_BASE_URL", top_k=2)

    assert results
    assert results[0].segment_id == "seg_0"
    assert results[0].evidence_type == "speech"


def test_bm25_store_filters_frame_caption_evidence() -> None:
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=10,
            transcript_text="No visual detail here.",
            visual_captions=[
                FrameCaption(frame_id="f0", timestamp=2, caption="Screen shows a Save button.", model="test")
            ],
        )
    ]

    results = BM25Store("v1", segments).search("Save button", top_k=5, evidence_types=["frame_caption"])

    assert results
    assert results[0].evidence_type == "frame_caption"
    assert results[0].frame_id == "f0"


def test_reciprocal_rank_fusion_deduplicates_and_combines_ranks() -> None:
    vector = [
        _evidence("v1:seg_0:speech", "seg_0", 0.8),
        _evidence("v1:seg_1:speech", "seg_1", 0.7),
    ]
    bm25 = [
        _evidence("v1:seg_1:speech", "seg_1", 1.0),
        _evidence("v1:seg_2:speech", "seg_2", 0.9),
    ]

    fused = reciprocal_rank_fusion(vector, bm25, top_k=3, rrf_k=60)

    assert [item.evidence_id for item in fused].count("v1:seg_1:speech") == 1
    assert fused[0].evidence_id == "v1:seg_1:speech"
    assert all(0.0 <= item.score <= 1.0 for item in fused)


def test_hybrid_retriever_uses_bm25_when_vector_has_no_match() -> None:
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=10,
            transcript_text="The exact identifier GLM-4.6 appears here.",
        )
    ]
    retriever = HybridEvidenceRetriever(_EmptyVectorStore(), Settings(_env_file=None, RETRIEVAL_MODE="hybrid"))

    results = retriever.search("v1", "GLM-4.6", segments, top_k=3)

    assert results
    assert results[0].segment_id == "seg_0"


def test_conversation_agent_search_for_intent_can_use_hybrid_bm25(tmp_path) -> None:
    segments = [
        MultimodalSegment(
            segment_id="seg_0",
            start=0,
            end=10,
            transcript_text="The exact identifier CHROMA_PATH is described here.",
        )
    ]
    agent = ConversationAgent("v1", tmp_path, settings=Settings(_env_file=None, RETRIEVAL_MODE="hybrid"))
    agent.store = _EmptyVectorStore()
    agent.retriever = HybridEvidenceRetriever(agent.store, agent.settings)

    intent = _intent("CHROMA_PATH")
    results = agent._search_for_intent(intent, top_k=3, segments=segments)

    assert results
    assert results[0].segment_id == "seg_0"


def test_tokenize_adds_chinese_bigrams() -> None:
    tokens = tokenize("证据绑定")
    assert "证据" in tokens
    assert "绑定" in tokens


def _evidence(evidence_id: str, segment_id: str, score: float) -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        video_id="v1",
        segment_id=segment_id,
        evidence_type="speech",
        text=f"text {segment_id}",
        start=0,
        end=1,
        score=score,
    )


def _intent(query: str):
    from app.models import ResolvedIntent

    return ResolvedIntent(
        original_question=query,
        rewritten_question=query,
        question_type="factual",
        retrieval_queries=[query],
        required_evidence_types=["speech"],
        reason="test",
    )


class _EmptyVectorStore:
    def search(self, video_id: str, query: str, top_k: int = 8, evidence_types=None):
        return []
