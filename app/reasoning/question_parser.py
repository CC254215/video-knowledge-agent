from __future__ import annotations

import re

from app.models import DeterministicQuestionContext, MultimodalSegment, RuleHints, TimeConstraint, VideoMetadata


SILENT_PLACEHOLDER = "silent video. no spoken transcript is available"

_CLOCK_RE = re.compile(r"(?<!\d)(?:(\d{1,2})[:：])?(\d{1,2})[:：](\d{2})(?!\d)")
_UNIT_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(分钟|秒|minutes?|seconds?)", re.I)
_RANGE_RE = re.compile(
    r"(?:从|from\s+)?(?P<start>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)\s*(?:到|至|[-–—]|to)\s*"
    r"(?P<end>\d{1,2}[:：]\d{2}(?:[:：]\d{2})?)",
    re.I,
)
_CHOICE_RE = re.compile(r"^\s*([A-H])\s*[.、):：]\s*(.+?)\s*$", re.M)
_WHOLE_VIDEO_RE = re.compile(
    r"整个视频|整段视频|全片|全程|贯穿视频|\b(?:entire|whole)\s+(?:video|clip)\b|"
    r"\bthroughout\s+the\s+(?:video|clip)\b|\boverall\s+process\s+(?:shown\s+)?in\s+the\s+video\b",
    re.I,
)
_EXPLICIT_VISUAL_RE = re.compile(r"画面里|画面中|图中|屏幕上|截图|按钮|界面上|视频里这一步|\b(?:on[- ]screen|in the (?:image|frame)|visually shown)\b", re.I)
_EXPLICIT_SPEECH_RE = re.compile(r"说了什么|讲了什么|提到什么|原话|语音|台词|转录|\b(?:say|said|mention|speech|spoken|transcript|quote)\b", re.I)
_RELATION_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("before_after", re.compile(r"之前|之后|以前|以后|\b(?:before|after)\b", re.I)),
    ("sequence", re.compile(r"先.+(?:再|然后|还是)|先后|顺序|\b(?:first|then|in what order)\b", re.I | re.S)),
    ("stage", re.compile(r"阶段|分为.+步|\b(?:phase|stage)s?\b", re.I)),
    ("transition", re.compile(r"转变|转折|过渡|\b(?:transition|turning point)s?\b", re.I)),
)


def parse_question_context(
    question: str,
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
) -> DeterministicQuestionContext:
    """Parse syntax and media facts only; semantic matches remain soft hints."""
    intervals = _parse_intervals(question)
    interval_spans = [(match.start(), match.end()) for match in _RANGE_RE.finditer(question)]
    timestamps = [
        TimeConstraint(start=seconds, end=seconds, source_text=source)
        for seconds, source, span in _parse_clock_and_unit_times(question)
        if not any(left <= span[0] and span[1] <= right for left, right in interval_spans)
    ]
    choices = [f"{label}. {text.strip()}" for label, text in _CHOICE_RE.findall(question)]
    real_transcript = [
        segment.transcript_text.strip()
        for segment in segments
        if segment.transcript_text.strip() and SILENT_PLACEHOLDER not in segment.transcript_text.lower()
    ]
    has_silent_marker = any(SILENT_PLACEHOLDER in segment.transcript_text.lower() for segment in segments)
    possible_relation = "none"
    for relation, pattern in _RELATION_HINTS:
        if pattern.search(question):
            possible_relation = relation
            break
    visual_hint = bool(_EXPLICIT_VISUAL_RE.search(question))
    speech_hint = bool(_EXPLICIT_SPEECH_RE.search(question))
    return DeterministicQuestionContext(
        video_title=metadata.title,
        video_duration=metadata.duration,
        has_real_speech=bool(real_transcript),
        has_audio=True if real_transcript else False if has_silent_marker else None,
        choice_based=bool(choices),
        explicit_whole_video=bool(_WHOLE_VIDEO_RE.search(question)),
        choices=choices,
        explicit_timestamps=_dedupe_constraints(timestamps),
        explicit_intervals=_dedupe_constraints(intervals),
        transcript_preview="\n".join(real_transcript[:6])[:1800],
        rule_hints=RuleHints(
            possible_temporal_relation=possible_relation,  # type: ignore[arg-type]
            explicit_visual_reference=visual_hint,
            explicit_speech_reference=speech_hint,
            confidence="medium" if visual_hint or speech_hint else "low",
        ),
    )


def has_real_speech(segments: list[MultimodalSegment]) -> bool:
    return any(segment.transcript_text.strip() and SILENT_PLACEHOLDER not in segment.transcript_text.lower() for segment in segments)


def _parse_intervals(question: str) -> list[TimeConstraint]:
    output: list[TimeConstraint] = []
    for match in _RANGE_RE.finditer(question):
        start, end = _clock_to_seconds(match.group("start")), _clock_to_seconds(match.group("end"))
        if start is not None and end is not None and end >= start:
            output.append(TimeConstraint(start=start, end=end, source_text=match.group(0)))
    return output


def _parse_clock_and_unit_times(question: str) -> list[tuple[float, str, tuple[int, int]]]:
    output: list[tuple[float, str, tuple[int, int]]] = []
    for match in _CLOCK_RE.finditer(question):
        seconds = _clock_to_seconds(match.group(0))
        if seconds is not None:
            output.append((seconds, match.group(0), match.span()))
    for match in _UNIT_TIME_RE.finditer(question):
        value, unit = float(match.group(1)), match.group(2).lower()
        output.append((value * 60.0 if unit.startswith("分钟") or unit.startswith("minute") else value, match.group(0), match.span()))
    return output


def _clock_to_seconds(value: str) -> float | None:
    parts = re.split(r"[:：]", value.strip())
    if len(parts) not in {2, 3} or not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    return float(numbers[0] * 60 + numbers[1]) if len(numbers) == 2 else float(numbers[0] * 3600 + numbers[1] * 60 + numbers[2])


def _dedupe_constraints(items: list[TimeConstraint]) -> list[TimeConstraint]:
    output: list[TimeConstraint] = []
    seen: set[tuple[float, float]] = set()
    for item in items:
        key = (round(item.start, 3), round(item.end, 3))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
