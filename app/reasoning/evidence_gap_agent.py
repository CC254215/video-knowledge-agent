from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models import EvidenceGapDecision, MultimodalSegment, RefineRange, RetrievedEvidence, VideoMetadata
from app.reasoning.summarizer import format_timestamp


TIME_WINDOW_SECONDS = 60.0


@dataclass
class TimeAnchor:
    timestamp: float
    text: str


class EvidenceGapAgent:
    """Rule-first evidence gap decision module.

    The agent is intentionally conservative: if a question could plausibly be
    about the video, it remains video-relevant and proceeds to evidence review.
    """

    def decide(
        self,
        question: str,
        metadata: VideoMetadata,
        segments: list[MultimodalSegment],
        evidence: list[RetrievedEvidence],
        agreement: float,
        confidence: str,
        needs_visual: bool,
    ) -> EvidenceGapDecision:
        relevance = judge_video_relevance(question, metadata, segments)
        if not relevance["is_video_relevant"]:
            return EvidenceGapDecision(
                is_video_relevant=False,
                relevance_reason=relevance["relevance_reason"],
                early_exit_reply=relevance["early_exit_reply"],
                should_refine=False,
                reasons=["not_video_relevant"],
                confidence=0.9,
                round_traces=["round0:not_video_relevant"],
            )

        anchors = extract_time_anchors(question, metadata.duration)
        review = review_existing_evidence(question, anchors, evidence, needs_visual)
        should_refine = bool(review["gaps"]) and (agreement < 0.75 or confidence == "low" or anchors or needs_visual)
        target_ranges = build_target_ranges(anchors, evidence, segments, review["gaps"])
        reasons = list(review["gaps"])
        if agreement < 0.3:
            reasons.append("low_agreement")
        if confidence == "low":
            reasons.append("low_confidence")
        return EvidenceGapDecision(
            is_video_relevant=True,
            relevance_reason=relevance["relevance_reason"],
            early_exit_reply=None,
            should_refine=should_refine and bool(target_ranges),
            reasons=_unique(reasons),
            target_ranges=target_ranges if should_refine else [],
            confidence=max(0.0, min(1.0, agreement)),
            round_traces=[
                f"round0:relevant:{relevance['relevance_reason']}",
                f"round1:time_anchors={[anchor.text for anchor in anchors]}",
                f"round2:gaps={review['gaps']}",
                f"round3:should_refine={should_refine}",
            ],
        )


def judge_video_relevance(question: str, metadata: VideoMetadata, segments: list[MultimodalSegment]) -> dict[str, Any]:
    q = question.strip()
    if not q:
        return {
            "is_video_relevant": False,
            "relevance_reason": "empty_question",
            "early_exit_reply": "请先输入一个和当前视频相关的问题。",
        }
    if q in {"你好", "谢谢", "hello", "hi"}:
        return {
            "is_video_relevant": False,
            "relevance_reason": "chat_smalltalk",
            "early_exit_reply": "这个问题和当前视频内容没有直接关系。你可以继续问我视频里的观点、时间点或证据。",
        }
    video_text = f"{metadata.title} {' '.join(segment.transcript_text[:400] for segment in segments[:8])}".lower()
    question_terms = [term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]{2,}", q)]
    video_terms = set(re.findall(r"[\w\u4e00-\u9fff]{2,}", video_text))
    if any(term in video_text for term in question_terms):
        return {"is_video_relevant": True, "relevance_reason": "question_terms_overlap_video_metadata_or_transcript", "early_exit_reply": None}
    if _looks_video_question(q):
        return {"is_video_relevant": True, "relevance_reason": "question_asks_about_video_content_or_time", "early_exit_reply": None}
    if len(question_terms) <= 2 and not (set(question_terms) & video_terms):
        return {
            "is_video_relevant": False,
            "relevance_reason": "no_overlap_with_video_topic",
            "early_exit_reply": "这个问题和当前视频内容没有直接关系。当前视频证据不足以回答它；如果你愿意，我可以把它作为通用问题单独解释。",
        }
    return {"is_video_relevant": True, "relevance_reason": "conservative_relevance_possible", "early_exit_reply": None}


def extract_time_anchors(question: str, duration: float | None = None) -> list[TimeAnchor]:
    anchors: list[TimeAnchor] = []
    for match in re.finditer(r"(\d{1,2})[:：](\d{1,2})(?:[:：](\d{1,2}))?", question):
        parts = [int(part) for part in match.groups(default="0")]
        timestamp = parts[0] * 60 + parts[1] if match.group(3) is None else parts[0] * 3600 + parts[1] * 60 + parts[2]
        anchors.append(TimeAnchor(float(timestamp), match.group(0)))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*分钟", question):
        anchors.append(TimeAnchor(float(match.group(1)) * 60.0, match.group(0)))
    for match in re.finditer(r"([一二两三四五六七八九十]+)\s*分钟", question):
        value = _chinese_number(match.group(1))
        if value is not None:
            anchors.append(TimeAnchor(float(value) * 60.0, match.group(0)))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*秒", question):
        anchors.append(TimeAnchor(float(match.group(1)), match.group(0)))
    if "开头" in question:
        anchors.append(TimeAnchor(0.0, "开头"))
    if "结尾" in question and duration:
        anchors.append(TimeAnchor(max(0.0, duration - 60.0), "结尾"))
    return _dedupe_anchors(anchors)


def review_existing_evidence(
    question: str,
    anchors: list[TimeAnchor],
    evidence: list[RetrievedEvidence],
    needs_visual: bool,
) -> dict[str, Any]:
    gaps: list[str] = []
    if anchors:
        for anchor in anchors:
            if not any(item.start - 5 <= anchor.timestamp <= item.end + 5 for item in evidence):
                gaps.append(f"missing_time_anchor:{format_timestamp(anchor.timestamp)}")
    if not evidence:
        gaps.append("no_retrieved_evidence")
    if needs_visual and not any(item.evidence_type == "frame_caption" and item.text.strip() for item in evidence):
        gaps.append("missing_frame_caption")
    if _asks_action_items(question) and not any(item.evidence_type == "speech" and item.text.strip() for item in evidence):
        gaps.append("missing_speech_for_action_items")
    return {"gaps": _unique(gaps)}


def build_target_ranges(
    anchors: list[TimeAnchor],
    evidence: list[RetrievedEvidence],
    segments: list[MultimodalSegment],
    gaps: list[str],
) -> list[RefineRange]:
    ranges: list[RefineRange] = []
    for anchor in anchors:
        start = max(0.0, anchor.timestamp - TIME_WINDOW_SECONDS)
        end = anchor.timestamp + TIME_WINDOW_SECONDS
        ranges.append(RefineRange(start=start, end=end, reason=f"time_anchor:{anchor.text}", priority=1))
    if not ranges and evidence:
        for item in evidence[:2]:
            ranges.append(RefineRange(start=max(0.0, item.start - 15), end=item.end + 15, reason="retrieved_evidence_gap", priority=2))
    if not ranges and segments:
        segment = segments[0]
        ranges.append(RefineRange(start=segment.start, end=segment.end, reason="fallback_first_segment", priority=3))
    return _merge_ranges(ranges)[:3]


def _asks_action_items(question: str) -> bool:
    return any(token in question for token in ("操作建议", "操作上的建议", "实操建议", "行动建议", "风控建议", "交易建议", "建议"))


def _looks_video_question(question: str) -> bool:
    return any(
        token in question
        for token in (
            "视频",
            "作者",
            "这段",
            "这里",
            "里面",
            "提到",
            "讲了",
            "总结",
            "核心观点",
            "十分钟",
            "分钟",
            "秒",
        )
    )


def _dedupe_anchors(anchors: list[TimeAnchor]) -> list[TimeAnchor]:
    output: list[TimeAnchor] = []
    seen: set[int] = set()
    for anchor in anchors:
        key = round(anchor.timestamp)
        if key not in seen:
            output.append(anchor)
            seen.add(key)
    return output


def _merge_ranges(ranges: list[RefineRange]) -> list[RefineRange]:
    merged: list[RefineRange] = []
    for item in sorted(ranges, key=lambda row: (row.start, row.end)):
        if not merged or item.start > merged[-1].end + 2:
            merged.append(item)
        else:
            previous = merged[-1]
            merged[-1] = RefineRange(
                start=previous.start,
                end=max(previous.end, item.end),
                reason=f"{previous.reason};{item.reason}",
                priority=min(previous.priority, item.priority),
            )
    return merged


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _chinese_number(value: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value == "十":
        return 10
    if value.startswith("十"):
        return 10 + digits.get(value[1:], 0)
    if "十" in value:
        left, _, right = value.partition("十")
        return digits.get(left, 0) * 10 + digits.get(right, 0)
    return digits.get(value)
