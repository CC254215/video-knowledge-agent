from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app import pipeline
from app.config import Settings, get_settings
from app.evals.metrics import mean_best_temporal_iou, precision_recall_f1, reciprocal_rank, token_f1
from app.models import ConversationTurn, MultimodalSegment, RetrievedEvidence
from app.reasoning.conversation_agent import ConversationAgent
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.retrieval.hybrid_retriever import HybridEvidenceRetriever


class EvalCase(BaseModel):
    case_id: str | None = None
    video_id: str
    question: str
    gold_answers: list[str] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)
    gold_evidence_ids: list[str] = Field(default_factory=list)
    gold_segment_ids: list[str] = Field(default_factory=list)
    gold_time_ranges: list[tuple[float, float]] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    should_abstain: bool = False


@dataclass
class EvalOptions:
    top_k: int = 5
    run_answers: bool = False
    fail_on_threshold: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


DEFAULT_THRESHOLDS = {
    "retrieval_hit_rate": 0.70,
    "segment_recall": 0.60,
    "grounding_precision": 0.80,
    "answer_contains_recall": 0.60,
}


def load_eval_cases(path: Path) -> list[EvalCase]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    return [_case_from_row(row, index) for index, row in enumerate(rows)]


def evaluate_dataset(
    dataset_path: Path,
    settings: Settings | None = None,
    options: EvalOptions | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    options = options or EvalOptions()
    cases = load_eval_cases(dataset_path)
    if not cases:
        return {
            "dataset_path": str(dataset_path),
            "case_count": 0,
            "options": {"top_k": options.top_k, "run_answers": options.run_answers},
            "aggregate": {},
            "thresholds": {},
            "ok": False,
            "error": "empty_dataset",
            "cases": [],
        }
    case_results = [evaluate_case(case, settings, options) for case in cases]
    aggregate = _aggregate(case_results)
    thresholds = _threshold_report(aggregate, DEFAULT_THRESHOLDS)
    return {
        "dataset_path": str(dataset_path),
        "case_count": len(cases),
        "options": {"top_k": options.top_k, "run_answers": options.run_answers},
        "aggregate": aggregate,
        "thresholds": thresholds,
        "ok": all(item["passed"] for item in thresholds.values()) if thresholds else True,
        "cases": case_results,
    }


def evaluate_case(case: EvalCase, settings: Settings, options: EvalOptions) -> dict[str, Any]:
    started = time.monotonic()
    segments = pipeline.read_multimodal_segments(case.video_id, settings)
    if not segments:
        segments = pipeline.ensure_multimodal_segments(case.video_id, settings)
    store = ChromaMemoryStore(settings, use_persistent=False)
    store.add_multimodal_segments(case.video_id, segments)
    retrieved = HybridEvidenceRetriever(store, settings).search(
        case.video_id,
        case.question,
        segments,
        top_k=options.top_k,
        evidence_types=case.evidence_types or ["speech", "frame_caption"],
    )
    turn = (
        ConversationAgent(case.video_id, pipeline.video_dir(case.video_id, settings), settings=settings, store=store).answer(
            case.question,
            conversation_id=f"eval:{case.case_id}",
            persist=False,
            allow_refinement=True,
            persist_trace=True,
            persist_refinement=False,
        )
        if options.run_answers
        else None
    )
    return score_case(case, retrieved, segments, turn, latency_seconds=time.monotonic() - started)


def score_case(
    case: EvalCase,
    retrieved: list[RetrievedEvidence],
    segments: list[MultimodalSegment],
    turn: ConversationTurn | None = None,
    latency_seconds: float = 0.0,
) -> dict[str, Any]:
    gold_segment_ids = _gold_segment_ids(case)
    retrieved_segment_ids = _unique([item.segment_id for item in retrieved])
    retrieved_evidence_ids = [item.evidence_id for item in retrieved]
    retrieved_ranges = [(item.start, item.end) for item in retrieved]
    segment_scores = precision_recall_f1(retrieved_segment_ids, gold_segment_ids)
    evidence_scores = precision_recall_f1(retrieved_evidence_ids, case.gold_evidence_ids)
    answer = turn.answer if turn else ""
    grounding_scores = precision_recall_f1(_unique(turn.evidence_ids if turn else []), gold_segment_ids)
    valid_evidence_id_set = _valid_evidence_ids(case.video_id, segments)
    cited_evidence_ids = _cited_evidence_ids(turn)
    invalid_citations = [item for item in cited_evidence_ids if item not in valid_evidence_id_set and item not in gold_segment_ids]
    metrics = {
        "retrieval_hit": 1.0 if set(retrieved_segment_ids) & set(gold_segment_ids) else 0.0,
        "retrieval_mrr": reciprocal_rank(retrieved_segment_ids, gold_segment_ids),
        "segment_precision": segment_scores["precision"],
        "segment_recall": segment_scores["recall"],
        "segment_f1": segment_scores["f1"],
        "evidence_precision": evidence_scores["precision"],
        "evidence_recall": evidence_scores["recall"],
        "temporal_iou": mean_best_temporal_iou(retrieved_ranges, case.gold_time_ranges),
        "latency_seconds": latency_seconds,
    }
    if turn is not None:
        metrics.update(
            {
                "grounding_precision": grounding_scores["precision"],
                "grounding_recall": grounding_scores["recall"],
                "answer_contains_recall": _contains_recall(answer, case.answer_contains),
                "answer_token_f1": _best_answer_token_f1(answer, case.gold_answers),
                "abstention_accuracy": 1.0 if ("证据不足" in answer) == case.should_abstain else 0.0,
                "invalid_citation_count": float(len(invalid_citations)),
            }
        )
    return {
        "case_id": case.case_id,
        "video_id": case.video_id,
        "question": case.question,
        "retrieved_evidence_ids": retrieved_evidence_ids,
        "retrieved_segment_ids": retrieved_segment_ids,
        "answer": answer,
        "cited_segment_ids": _unique(turn.evidence_ids if turn else []),
        "metrics": metrics,
    }


def write_eval_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output_path)


def _case_from_row(row: dict[str, Any], index: int) -> EvalCase:
    normalized = dict(row)
    normalized.setdefault("case_id", str(row.get("id") or row.get("qid") or f"case_{index:04d}"))
    normalized["gold_answers"] = _string_list(row.get("gold_answers") or row.get("answers") or row.get("answer"))
    normalized["answer_contains"] = _string_list(row.get("answer_contains") or row.get("must_include"))
    normalized["gold_evidence_ids"] = _string_list(row.get("gold_evidence_ids") or row.get("evidence_ids"))
    normalized["gold_segment_ids"] = _string_list(row.get("gold_segment_ids") or row.get("segment_ids"))
    normalized["gold_time_ranges"] = _time_ranges(row.get("gold_time_ranges") or row.get("time_ranges"))
    return EvalCase.model_validate(normalized)


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _time_ranges(value: object) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    ranges: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        try:
            ranges.append((float(start), float(end)))
        except (TypeError, ValueError):
            continue
    return ranges


def _gold_segment_ids(case: EvalCase) -> list[str]:
    result = list(case.gold_segment_ids)
    for evidence_id in case.gold_evidence_ids:
        parts = evidence_id.split(":")
        if len(parts) >= 3 and parts[1] not in result:
            result.append(parts[1])
    return result


def _contains_recall(answer: str, required: list[str]) -> float:
    if not required:
        return 1.0
    lowered = answer.lower()
    return sum(1 for item in required if item.lower() in lowered) / len(required)


def _best_answer_token_f1(answer: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 1.0
    return max(token_f1(answer, gold) for gold in gold_answers)


def _valid_evidence_ids(video_id: str, segments: list[MultimodalSegment]) -> set[str]:
    valid: set[str] = set()
    for segment in segments:
        if segment.transcript_text.strip():
            valid.add(f"{video_id}:{segment.segment_id}:speech")
        for caption in segment.visual_captions:
            if caption.caption.strip():
                valid.add(f"{video_id}:{segment.segment_id}:frame_caption:{caption.frame_id}")
    return valid


def _cited_evidence_ids(turn: ConversationTurn | None) -> list[str]:
    if not turn:
        return []
    result = [str(item.get("evidence_id")) for item in turn.evidence if item.get("evidence_id")]
    result.extend(turn.query_evidence_ids)
    result.extend(turn.answer_evidence_ids)
    return _unique(result)


def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, float]:
    if not case_results:
        return {}
    metric_names = sorted({name for result in case_results for name in result["metrics"]})
    aggregate: dict[str, float] = {}
    for name in metric_names:
        values = [float(result["metrics"][name]) for result in case_results if name in result["metrics"]]
        aggregate[name] = sum(values) / len(values) if values else 0.0
    aggregate["retrieval_hit_rate"] = aggregate.pop("retrieval_hit", 0.0)
    return aggregate


def _threshold_report(aggregate: dict[str, float], thresholds: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    return {
        name: {"value": aggregate.get(name, 0.0), "minimum": minimum, "passed": aggregate.get(name, 0.0) >= minimum}
        for name, minimum in thresholds.items()
        if name in aggregate
    }


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
