from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.models import (
    ActionItem,
    AnalysisItem,
    CanonicalEntity,
    ConfidenceLevel,
    ModalityProfile,
    MultimodalSegment,
    OutlineItem,
    QuoteItem,
    SummaryReport,
    TranscriptSegment,
    VideoMetadata,
)
from app.reasoning.llm_client import LLMClient


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _shorten(text: str, max_chars: int = 180) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 1].rstrip() + "..."


def summarize_multimodal_video(
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    modality_profile: ModalityProfile,
    settings: Settings | None = None,
    log_dir: Path | None = None,
) -> SummaryReport:
    settings = settings or get_settings()
    if not segments:
        return _empty_summary(metadata, modality_profile, ["No multimodal segments available."])

    if settings.has_llm_config:
        try:
            client = LLMClient(settings, log_dir)
            if len(segments) > settings.summary_max_segments:
                return _map_reduce_summary(metadata, segments, modality_profile, settings, client)
            payload = client.generate_json(_summary_prompt(metadata, segments, modality_profile, settings), schema_hint=_summary_schema_hint())
            report = _coerce_llm_summary(metadata, segments, modality_profile, payload)
            report.summary_strategy = "single_pass"
            report.num_chunks = 1
            return report
        except Exception as exc:  # noqa: BLE001
            if settings.strict_runtime:
                raise RuntimeError(f"LLM summary failed in strict mode: {exc}") from exc
            fallback = _fallback_multimodal_summary(metadata, segments, modality_profile)
            fallback.warnings.append(f"LLM summary failed; used local fallback: {exc}")
            fallback.warnings.append("degraded_fallback")
            fallback.confidence = ConfidenceLevel.low
            fallback.generation_status = "degraded_fallback"
            return fallback

    fallback = _fallback_multimodal_summary(metadata, segments, modality_profile)
    fallback.warnings.append("Missing LLM config; used local grounded fallback summary.")
    fallback.warnings.append("degraded_fallback")
    fallback.confidence = ConfidenceLevel.low
    fallback.generation_status = "degraded_fallback"
    return fallback


def _map_reduce_summary(
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    modality_profile: ModalityProfile,
    settings: Settings,
    client: LLMClient,
) -> SummaryReport:
    chunks = _chunk_segments(segments, max(1, min(12, settings.summary_max_segments // 2 or 8)))
    chunk_payloads: list[dict[str, Any]] = []
    concurrency = max(1, min(len(chunks) or 1, settings.summary_map_concurrency))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_generate_chunk_summary, metadata, chunk, index, len(chunks), settings, client.log_dir): index
            for index, chunk in enumerate(chunks)
        }
        ordered: list[dict[str, Any] | None] = [None] * len(chunks)
        for future in as_completed(futures):
            index = futures[future]
            chunk = chunks[index]
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                payload = _fallback_chunk_summary(chunk, index, exc)
            payload["chunk_id"] = f"chunk_{index:04d}"
            payload["segment_ids"] = [segment.segment_id for segment in chunk]
            ordered[index] = payload
        chunk_payloads = [payload for payload in ordered if payload is not None]
    reduced = client.generate_json(_reduce_summary_prompt(metadata, chunk_payloads, modality_profile), schema_hint=_summary_schema_hint())
    report = _coerce_llm_summary(metadata, segments, modality_profile, reduced)
    report.summary_strategy = "map_reduce"
    report.num_chunks = len(chunks)
    report.chunk_summary_ids = [str(item.get("chunk_id")) for item in chunk_payloads]
    report.evidence_coverage.setdefault("chunk_summaries", chunk_payloads)
    return report


def _generate_chunk_summary(
    metadata: VideoMetadata,
    chunk: list[MultimodalSegment],
    index: int,
    total: int,
    settings: Settings,
    log_dir: Path | None,
) -> dict[str, Any]:
    client = LLMClient(settings, log_dir)
    attempts = 2
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return client.generate_json(
                _chunk_summary_prompt(metadata, chunk, index, total, settings),
                schema_hint=_chunk_summary_schema_hint(),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error or RuntimeError("chunk summary failed")


def _fallback_chunk_summary(chunk: list[MultimodalSegment], index: int, exc: Exception) -> dict[str, Any]:
    # Local chunk fallback is scoped to the map phase only. The reduce phase
    # still runs over all chunk payloads, and warnings make degradation auditable.
    cited = [segment.segment_id for segment in chunk if segment.transcript_text.strip()]
    claims = [
        {
            "claim": _shorten(segment.transcript_text, 140),
            "segment_ids": [segment.segment_id],
            "evidence_types": ["speech"],
        }
        for segment in chunk[:3]
        if segment.transcript_text.strip()
    ]
    return {
        "chunk_overview": _shorten(" ".join(segment.transcript_text for segment in chunk), 240),
        "local_claims": claims,
        "local_evidence": [
            {"segment_id": segment.segment_id, "evidence_type": "speech", "quote_or_caption": _shorten(segment.transcript_text, 160)}
            for segment in chunk[:4]
            if segment.transcript_text.strip()
        ],
        "local_open_questions": [],
        "local_action_items": [],
        "cited_segment_ids": cited,
        "warnings": [f"chunk_{index:04d} map summary fallback: {exc}"],
    }


def _chunk_segments(segments: list[MultimodalSegment], chunk_size: int) -> list[list[MultimodalSegment]]:
    return [segments[index : index + chunk_size] for index in range(0, len(segments), chunk_size)]


def _chunk_summary_prompt(metadata: VideoMetadata, chunk: list[MultimodalSegment], index: int, total: int, settings: Settings) -> str:
    block = "\n\n".join(_format_segment(segment, settings.summary_segment_chars, settings.summary_caption_chars) for segment in chunk)
    return f"""
You are summarizing chunk {index + 1}/{total} of a long video.
Use only the provided multimodal segments. OCR 已禁用。
Preserve exact segment_id values in every claim.
Mention whether evidence is speech or frame_caption when relevant.

video_title={metadata.title}

segments:
{block}

Return JSON only.
"""


def _chunk_summary_schema_hint() -> str:
    return """
{
  "chunk_overview": "...",
  "local_claims": [{"claim":"...", "segment_ids":["seg_0000"], "evidence_types":["speech","frame_caption"]}],
  "local_evidence": [{"segment_id":"seg_0000", "evidence_type":"speech", "quote_or_caption":"..."}],
  "local_open_questions": ["..."],
  "local_action_items": [{"action":"...", "segment_ids":["seg_0000"]}],
  "cited_segment_ids": ["seg_0000"]
}
"""


def _reduce_summary_prompt(metadata: VideoMetadata, chunk_payloads: list[dict[str, Any]], modality_profile: ModalityProfile) -> str:
    return f"""
You are reducing chunk summaries into a global Video Knowledge Agent summary.
Use only the chunk summaries below. Keep original segment_id citations.
Do not invent content. OCR 已禁用。Allowed evidence types: speech, frame, frame_caption.

video_title={metadata.title}
modality={modality_profile.mode.value}: {modality_profile.reason}

chunk_summaries:
{json_dumps(chunk_payloads)}

Return global JSON with quick_overview, structured_outline, deep_analysis, action_items,
important_quotes, open_questions, evidence_coverage.
"""


def summarize_video(
    metadata: VideoMetadata,
    segments: list[TranscriptSegment],
    modality_profile: ModalityProfile,
) -> SummaryReport:
    multimodal = [
        MultimodalSegment(
            segment_id=segment.segment_id,
            start=segment.start,
            end=segment.end,
            transcript_text=segment.text,
            evidence_types=["speech"],
            keywords=segment.keywords,
            entities=segment.entities,
        )
        for segment in segments
    ]
    return _fallback_multimodal_summary(metadata, multimodal, modality_profile)


def _summary_prompt(metadata: VideoMetadata, segments: list[MultimodalSegment], modality_profile: ModalityProfile, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    selected_segments = _select_summary_segments(segments, settings.summary_max_segments)
    segment_block = "\n\n".join(
        _format_segment(segment, settings.summary_segment_chars, settings.summary_caption_chars)
        for segment in selected_segments
    )
    return f"""
You are the summary module of Video Knowledge Agent.
Use ONLY the provided multimodal evidence. Do not invent facts.
Every key conclusion must cite existing segment_id values.
If evidence is insufficient, write insufficient_evidence for that item.
OCR is disabled. Allowed evidence types: speech, frame, frame_caption.
OCR 已禁用。
The evidence pack below is a coverage sample across the whole video, not a full transcript.
Build a chronological outline with explicit time ranges, not isolated moments.
For videos of at least 10 minutes, normally return 6-10 outline items, with no item spanning
more than about 3 minutes unless the evidence is genuinely one indivisible topic.
Every usable segment should be cited by at least one outline item.

Also extract only high-value named entities (software, products, models, libraries,
frameworks, companies, people, versions, abbreviations, or repeated ASR variants).
Do not rewrite ordinary sentences. Only canonicalize an ASR variant when title,
visual text, repeated context, or multiple independent mentions support it. Keep
uncertain candidates with needs_review=true and do not force a replacement.

Video metadata:
title={metadata.title}
author={metadata.author or ""}
source={metadata.source}
url={metadata.url or ""}
duration={metadata.duration}
modality={modality_profile.mode.value}: {modality_profile.reason}
segments_total={len(segments)}
segments_in_prompt={len(selected_segments)}

multimodal_segments:
{segment_block}

Return valid JSON with these fields:
quick_overview, structured_outline, deep_analysis, action_items, important_quotes,
open_questions, evidence_coverage, canonical_entities.
"""
    return f"""
你是 Video Knowledge Agent 的摘要模块。只能基于给定 multimodal_segments 生成摘要。

全局约束：
- 不要编造视频中没有的内容。
- 所有关键结论必须绑定 segment_id。
- 如果证据不足，输出 insufficient_evidence。
- 当前阶段 OCR 已禁用，不得使用或引用 OCR。
- 允许使用 speech 和 frame_caption；frame 只能表示关键帧存在和时间点。

视频 metadata:
title={metadata.title}
author={metadata.author or ""}
source={metadata.source}
url={metadata.url or ""}
duration={metadata.duration}
modality={modality_profile.mode.value}: {modality_profile.reason}

multimodal_segments:
{segment_block}

请输出 JSON，字段：
quick_overview, structured_outline, deep_analysis, action_items, important_quotes,
open_questions, evidence_coverage, canonical_entities。
"""


def _summary_schema_hint() -> str:
    return """
{
  "quick_overview": ["3 Chinese sentences"],
  "structured_outline": [
    {"timestamp":"[00:00:00] - [00:02:30]", "topic":"...", "key_points":["..."], "segment_ids":["seg_0000"]}
  ],
  "deep_analysis": [
    {"claim":"...", "evidence_segment_ids":["seg_0000"], "caveats":["..."]}
  ],
  "action_items": [
    {"action":"...", "evidence_segment_ids":["seg_0000"]}
  ],
  "important_quotes": [
    {"quote":"原文短句", "segment_id":"seg_0000", "timestamp":"[00:00:00]"}
  ],
  "open_questions": ["..."],
  "evidence_coverage": {"covered_segment_ids":["seg_0000"], "missing_evidence_items":[]}
  ,"canonical_entities": [
    {"canonical":"...", "aliases":["..."], "entity_type":"software", "confidence":0.9,
     "evidence_sources":["video_title","speech","frame_caption"], "reason":"...", "needs_review":false}
  ]
}
"""


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _coerce_llm_summary(
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    modality_profile: ModalityProfile,
    payload: dict[str, Any],
) -> SummaryReport:
    valid_ids = {segment.segment_id for segment in segments}
    warnings: list[str] = []

    outline: list[OutlineItem] = []
    for item in payload.get("structured_outline") or []:
        ids = _valid_ids(item.get("segment_ids"), valid_ids)
        if not ids:
            warnings.append(f"evidence_missing: outline topic={item.get('topic', '')}")
        _warn_invalid_ids(item.get("segment_ids"), valid_ids, warnings)
        outline.append(
            OutlineItem(
                timestamp=str(item.get("timestamp") or ""),
                topic=str(item.get("topic") or ""),
                key_points=[str(point) for point in item.get("key_points") or []],
                segment_ids=ids,
            )
        )

    analysis: list[AnalysisItem] = []
    for item in payload.get("deep_analysis") or []:
        ids = _valid_ids(item.get("evidence_segment_ids"), valid_ids)
        if not ids:
            warnings.append(f"evidence_missing: analysis claim={item.get('claim', '')}")
        _warn_invalid_ids(item.get("evidence_segment_ids"), valid_ids, warnings)
        analysis.append(
            AnalysisItem(
                claim=str(item.get("claim") or ""),
                evidence_segment_ids=ids,
                caveats=[str(caveat) for caveat in item.get("caveats") or []],
            )
        )

    actions: list[ActionItem] = []
    for item in payload.get("action_items") or []:
        ids = _valid_ids(item.get("evidence_segment_ids"), valid_ids)
        if not ids:
            warnings.append(f"evidence_missing: action={item.get('action', '')}")
        _warn_invalid_ids(item.get("evidence_segment_ids"), valid_ids, warnings)
        actions.append(ActionItem(action=str(item.get("action") or ""), evidence_segment_ids=ids))

    quotes: list[QuoteItem] = []
    for item in payload.get("important_quotes") or []:
        segment_id = str(item.get("segment_id") or "")
        if segment_id not in valid_ids:
            warnings.append(f"invalid_segment_id: quote segment_id={segment_id}")
            continue
        quotes.append(
            QuoteItem(
                quote=str(item.get("quote") or ""),
                segment_id=segment_id,
                timestamp=str(item.get("timestamp") or ""),
            )
        )

    if not outline and not analysis and not actions:
        warnings.append("evidence_missing: LLM summary did not provide grounded items.")

    confidence = ConfidenceLevel.low if any("evidence_missing" in warning for warning in warnings) else ConfidenceLevel.high
    entities: list[CanonicalEntity] = []
    for item in payload.get("canonical_entities") or []:
        if not isinstance(item, dict) or not str(item.get("canonical") or "").strip():
            continue
        try:
            entities.append(CanonicalEntity.model_validate(item))
        except Exception:
            continue
    report = SummaryReport(
        video_id=metadata.video_id,
        quick_overview=[str(item) for item in payload.get("quick_overview") or []][:3],
        structured_outline=outline,
        deep_analysis=analysis,
        action_items=actions,
        important_quotes=quotes,
        open_questions=[str(item) for item in payload.get("open_questions") or []],
        modality_note=f"{modality_profile.mode.value}: {modality_profile.reason}",
        evidence_coverage=payload.get("evidence_coverage") if isinstance(payload.get("evidence_coverage"), dict) else {},
        confidence=confidence,
        canonical_entities=entities,
        generation_status="llm_success",
        warnings=warnings,
    )
    _ensure_outline_time_coverage(report, segments)
    return report


def _fallback_multimodal_summary(
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    modality_profile: ModalityProfile,
) -> SummaryReport:
    outline = [
        OutlineItem(
            timestamp=format_timestamp(segment.start),
            topic=segment.keywords[0] if segment.keywords else f"片段 {segment.segment_id}",
            key_points=[_shorten(segment.transcript_text or segment.visual_summary, 220)],
            segment_ids=[segment.segment_id],
        )
        for segment in segments
        if segment.transcript_text.strip() or segment.visual_summary.strip()
    ]
    analysis = [
        AnalysisItem(
            claim=_shorten(segment.transcript_text or segment.visual_summary, 160),
            evidence_segment_ids=[segment.segment_id],
            caveats=["本地 fallback 只做证据绑定，不替代 LLM 深度分析。"],
        )
        for segment in segments[:6]
        if segment.transcript_text.strip() or segment.visual_summary.strip()
    ]
    quotes = [
        QuoteItem(quote=_shorten(segment.transcript_text, 160), segment_id=segment.segment_id, timestamp=format_timestamp(segment.start))
        for segment in segments[:8]
        if segment.transcript_text.strip()
    ]
    return SummaryReport(
        video_id=metadata.video_id,
        quick_overview=[
            f"《{metadata.title}》已生成基于视频证据的本地摘要。",
            f"当前摘要覆盖 {len(segments)} 个 multimodal segments。",
            f"证据类型限定为 speech、frame、frame_caption；OCR 已禁用。",
        ],
        structured_outline=outline,
        deep_analysis=analysis,
        action_items=[],
        important_quotes=quotes,
        open_questions=["哪些结论值得升级为 user_verified 长期知识？", "是否需要针对视觉问题补充关键帧描述？"],
        modality_note=f"{modality_profile.mode.value}: {modality_profile.reason}",
        evidence_coverage={"covered_segment_ids": [segment.segment_id for segment in segments], "mode": "fallback"},
        confidence=ConfidenceLevel.medium,
        generation_status="degraded_fallback",
        warnings=[],
    )


def _empty_summary(metadata: VideoMetadata, modality_profile: ModalityProfile, warnings: list[str]) -> SummaryReport:
    return SummaryReport(
        video_id=metadata.video_id,
        quick_overview=["当前视频没有可用证据，无法生成可靠摘要。"],
        structured_outline=[],
        deep_analysis=[],
        action_items=[],
        important_quotes=[],
        open_questions=["是否需要重新获取字幕或 ASR 转录？"],
        modality_note=modality_profile.reason,
        evidence_coverage={"covered_segment_ids": []},
        confidence=ConfidenceLevel.low,
        generation_status="degraded_fallback",
        warnings=warnings,
    )


def _select_summary_segments(segments: list[MultimodalSegment], limit: int) -> list[MultimodalSegment]:
    if len(segments) <= limit:
        return segments
    if limit <= 1:
        return [segments[0]]
    indexes = sorted({round(i * (len(segments) - 1) / (limit - 1)) for i in range(limit)})
    return [segments[index] for index in indexes]


def _ensure_outline_time_coverage(report: SummaryReport, segments: list[MultimodalSegment]) -> None:
    if not segments:
        return
    usable_segments = [segment for segment in segments if segment.transcript_text.strip() or segment.visual_summary.strip()]
    if not usable_segments:
        return
    valid_by_id = {segment.segment_id: segment for segment in usable_segments}
    covered_ids = {
        segment_id
        for item in report.structured_outline
        for segment_id in item.segment_ids
        if segment_id in valid_by_id
    }
    if not covered_ids:
        return
    covered_starts = [valid_by_id[segment_id].start for segment_id in covered_ids]
    latest_covered = max(covered_starts, default=0.0)
    latest_available = max(segment.start for segment in usable_segments)
    if latest_available <= 0 or latest_covered >= latest_available * 0.75:
        return
    added = 0
    for segment in usable_segments:
        if segment.start <= latest_covered or segment.segment_id in covered_ids:
            continue
        report.structured_outline.append(
            OutlineItem(
                timestamp=format_timestamp(segment.start),
                topic=segment.keywords[0] if segment.keywords else f"片段 {segment.segment_id}",
                key_points=[_shorten(segment.transcript_text or segment.visual_summary, 220)],
                segment_ids=[segment.segment_id],
            )
        )
        covered_ids.add(segment.segment_id)
        added += 1
    if added:
        coverage = report.evidence_coverage if isinstance(report.evidence_coverage, dict) else {}
        existing = coverage.get("covered_segment_ids")
        if isinstance(existing, list):
            coverage["covered_segment_ids"] = sorted({str(item) for item in existing} | covered_ids)
        report.evidence_coverage = coverage
        report.warnings.append(f"outline_time_coverage_patched: added_tail_items={added}")


def _format_segment(segment: MultimodalSegment, transcript_chars: int = 500, caption_chars: int = 300) -> str:
    captions = " ".join(caption.caption for caption in segment.visual_captions if caption.caption).strip()
    return (
        f"segment_id={segment.segment_id}\n"
        f"start={segment.start:.2f}\n"
        f"end={segment.end:.2f}\n"
        f"transcript_text={segment.transcript_text[:transcript_chars]}\n"
        f"representative_frame_ids={segment.representative_frame_ids}\n"
        f"visual_captions={(segment.visual_summary or captions)[:caption_chars]}\n"
        f"evidence_types={[item for item in segment.evidence_types if item in {'speech', 'frame', 'frame_caption'}]}"
    )


def _valid_ids(value: object, valid_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        segment_id = str(item)
        if segment_id in valid_ids and segment_id not in result:
            result.append(segment_id)
    return result


def _warn_invalid_ids(value: object, valid_ids: set[str], warnings: list[str]) -> None:
    if not isinstance(value, list):
        return
    for item in value:
        segment_id = str(item)
        if segment_id not in valid_ids:
            warnings.append(f"invalid_segment_id: {segment_id}")
