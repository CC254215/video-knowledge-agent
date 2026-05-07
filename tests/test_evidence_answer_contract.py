from app.models import ConfidenceLevel, QAAnswer
from app.reasoning.qa import answer_question
from app.retrieval.vector_store import InMemoryVectorStore
from app.transcript.segmenter import segment_transcript


def test_qaanswer_requires_evidence_and_timestamps():
    answer = QAAnswer(
        question="核心论点是什么？",
        answer="视频强调证据绑定。",
        evidence_segment_ids=["seg_0001"],
        timestamps=["[00:00:10]"],
        quotes=["每个结论必须绑定证据。"],
        confidence=ConfidenceLevel.high,
        needs_visual_check=False,
        reason="测试",
    )
    assert answer.evidence_segment_ids
    assert answer.timestamps


def test_local_qa_returns_evidence_contract():
    segments = segment_transcript(
        [
            {"start": 0, "end": 60, "text": "这个视频的核心论点是所有关键结论必须绑定时间戳证据。"},
            {"start": 60, "end": 120, "text": "问答回答需要引用 evidence segment，并计算一致性信号。"},
        ]
    )
    store = InMemoryVectorStore()
    store.add_segments("v1", segments)
    answer = answer_question("v1", "核心论点是什么？", store)
    assert answer.evidence_segment_ids
    assert answer.timestamps
    assert answer.confidence in {ConfidenceLevel.high, ConfidenceLevel.medium, ConfidenceLevel.low}
