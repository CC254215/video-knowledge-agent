from app.retrieval.dpp import greedy_multimodal_dpp_select


def test_multimodal_dpp_is_not_plain_relevance_topk_for_duplicates():
    ids = ["a", "b", "c"]
    relevance = [0.99, 0.98, 0.70]
    embeddings = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ]
    selected = greedy_multimodal_dpp_select(ids, relevance, embeddings, ["speech", "speech", "frame_caption"], k=2, min_relevance=0.0)
    assert selected != ["a", "b"]
    assert "c" in selected
