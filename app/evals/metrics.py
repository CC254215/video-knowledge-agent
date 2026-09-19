from app.retrieval.evidence import compute_evidence_agreement


def evidence_agreement_metric(query_ids: list[str], evidence_ids: list[str]) -> float:
    return compute_evidence_agreement(query_ids, evidence_ids)


def precision_recall_f1(predicted: list[str], gold: list[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def reciprocal_rank(ranked_ids: list[str], gold_ids: list[str]) -> float:
    gold_set = set(gold_ids)
    if not ranked_ids or not gold_set:
        return 0.0
    for index, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold_set:
            return 1.0 / index
    return 0.0


def token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = _tokens(predicted)
    gold_tokens = _tokens(gold)
    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token in predicted_tokens:
        count = gold_counts.get(token, 0)
        if count:
            overlap += 1
            gold_counts[token] = count - 1
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    intersection = max(0.0, end - start)
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def mean_best_temporal_iou(predicted: list[tuple[float, float]], gold: list[tuple[float, float]]) -> float:
    if not predicted or not gold:
        return 0.0
    scores = [max(temporal_iou(gold_range, predicted_range) for predicted_range in predicted) for gold_range in gold]
    return sum(scores) / len(scores)


def _tokens(text: str) -> list[str]:
    import re

    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)]
