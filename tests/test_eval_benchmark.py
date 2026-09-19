from __future__ import annotations

import json
from pathlib import Path

from app.evals.benchmark import EvalCase, EvalOptions, evaluate_dataset, load_eval_cases, score_case
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment, RetrievedEvidence


def test_load_eval_cases_accepts_jsonl_aliases(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "video_id": "v1",
                "question": "核心是什么？",
                "answer": "证据绑定",
                "evidence_ids": ["v1:seg_0000:speech"],
                "time_ranges": [{"start": 0, "end": 10}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_eval_cases(path)

    assert cases[0].case_id == "q1"
    assert cases[0].gold_answers == ["证据绑定"]
    assert cases[0].gold_time_ranges == [(0.0, 10.0)]


def test_score_case_reports_retrieval_grounding_and_answer_metrics() -> None:
    case = EvalCase(
        case_id="q1",
        video_id="v1",
        question="核心是什么？",
        gold_answers=["关键结论必须绑定时间戳证据"],
        answer_contains=["时间戳证据"],
        gold_evidence_ids=["v1:seg_0000:speech"],
        gold_time_ranges=[(0, 10)],
    )
    retrieved = [
        RetrievedEvidence(
            evidence_id="v1:seg_0000:speech",
            video_id="v1",
            segment_id="seg_0000",
            evidence_type="speech",
            text="关键结论必须绑定时间戳证据。",
            start=0,
            end=10,
            score=1.0,
        )
    ]
    turn = ConversationTurn(
        turn_id="t1",
        video_id="v1",
        user_question=case.question,
        answer="视频强调关键结论必须绑定时间戳证据。",
        evidence_ids=["seg_0000"],
        evidence=[{"evidence_id": "v1:seg_0000:speech"}],
        confidence=ConfidenceLevel.high,
        needs_visual_check=False,
        reason="test",
    )

    result = score_case(
        case,
        retrieved,
        [MultimodalSegment(segment_id="seg_0000", start=0, end=10, transcript_text="关键结论必须绑定时间戳证据。")],
        turn,
    )

    assert result["metrics"]["retrieval_hit"] == 1.0
    assert result["metrics"]["segment_recall"] == 1.0
    assert result["metrics"]["grounding_precision"] == 1.0
    assert result["metrics"]["answer_contains_recall"] == 1.0
    assert result["metrics"]["temporal_iou"] == 1.0


def test_evaluate_dataset_runs_retrieval_only(monkeypatch, tmp_path: Path) -> None:
    dataset = tmp_path / "cases.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "video_id": "v1",
                    "question": "气球实验发生了什么？",
                    "gold_segment_ids": ["seg_0000"],
                    "answer_contains": ["气球"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segment = MultimodalSegment(segment_id="seg_0000", start=0, end=10, transcript_text="气球实验里小气球的气流向大气球。")
    monkeypatch.setattr("app.pipeline.read_multimodal_segments", lambda video_id, settings: [segment])

    report = evaluate_dataset(dataset, options=EvalOptions(run_answers=False))

    assert report["case_count"] == 1
    assert report["aggregate"]["retrieval_hit_rate"] == 1.0
