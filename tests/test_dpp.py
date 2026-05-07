from app.retrieval.dpp import greedy_dpp_select
from app.retrieval.evidence import compute_evidence_agreement


def test_dpp_reduces_duplicate_evidence():
    selected = greedy_dpp_select(
        candidate_ids=["a", "b", "c"],
        relevance_scores=[0.95, 0.94, 0.75],
        embeddings=[
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        k=2,
        diversity_weight=0.8,
    )
    assert selected == ["a", "c"]


def test_evidence_agreement_formula():
    assert compute_evidence_agreement(["a", "b", "c"], ["b", "c", "d"]) == 2 / 3
    assert compute_evidence_agreement([], ["a"]) == 0.0
