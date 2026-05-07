from __future__ import annotations

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def greedy_dpp_select(
    candidate_ids: list[str],
    relevance_scores: list[float],
    embeddings: list[list[float]],
    k: int,
    diversity_weight: float = 0.7,
) -> list[str]:
    """Select relevant but diverse evidence.

    DPP is not a truth verifier. In this project it is only an evidence selection mechanism and
    a consistency signal for comparing query-selected and answer-selected evidence sets.
    """
    if k <= 0 or not candidate_ids:
        return []
    vectors = [np.array(vector, dtype=float) for vector in embeddings]
    selected: list[int] = []
    remaining = set(range(len(candidate_ids)))

    while remaining and len(selected) < k:
        best_index = None
        best_gain = -float("inf")
        for index in remaining:
            relevance = relevance_scores[index]
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(_cosine(vectors[index], vectors[chosen]) for chosen in selected)
            gain = relevance - diversity_weight * max(0.0, diversity_penalty)
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return [candidate_ids[index] for index in selected]


def greedy_multimodal_dpp_select(
    candidate_ids: list[str],
    relevance_scores: list[float],
    embeddings: list[list[float]],
    evidence_types: list[str],
    k: int,
    diversity_weight: float = 0.55,
    modality_bonus: float = 0.08,
    min_relevance: float = 0.05,
) -> list[str]:
    """Select multimodal evidence without treating DPP as a truth verifier.

    The modality bonus nudges the selector to avoid all-speech or all-frame results when similarly
    relevant evidence exists. It does not force low-relevance visual evidence into the final set.
    """
    if k <= 0 or not candidate_ids:
        return []
    vectors = [np.array(vector, dtype=float) for vector in embeddings]
    selected: list[int] = []
    remaining = {index for index, score in enumerate(relevance_scores) if score >= min_relevance}
    if not remaining:
        remaining = set(range(len(candidate_ids)))
    selected_types: set[str] = set()

    while remaining and len(selected) < k:
        best_index = None
        best_gain = -float("inf")
        for index in remaining:
            relevance = relevance_scores[index]
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(_cosine(vectors[index], vectors[chosen]) for chosen in selected)
            bonus = modality_bonus if evidence_types[index] not in selected_types and relevance >= min_relevance else 0.0
            gain = relevance + bonus - diversity_weight * max(0.0, diversity_penalty)
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
        selected_types.add(evidence_types[best_index])
        remaining.remove(best_index)
    return [candidate_ids[index] for index in selected]
