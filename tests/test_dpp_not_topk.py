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


def test_multimodal_dpp_prioritizes_speech_over_caption_when_close():
    selected = greedy_multimodal_dpp_select(
        ["caption", "speech"],
        [1.0, 0.9],
        [[1.0, 0.0], [1.0, 0.0]],
        ["frame_caption", "speech"],
        k=1,
        min_relevance=0.0,
    )
    assert selected == ["speech"]
