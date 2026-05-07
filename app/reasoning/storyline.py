from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.models import EvidenceStatus, MultimodalSegment, RetrievedEvidence, Storyline, StorylineNode, TranscriptSegment, VideoMetadata
from app.reasoning.llm_client import LLMClient
from app.reasoning.summarizer import format_timestamp
from app.retrieval.chroma_memory import ChromaMemoryStore

DEFAULT_STORYLINE_QUERY = "这个视频中有哪些对用户长期知识库有价值的观点、方法、事实、反驳和行动建议？"
BAD_TITLE_WORDS = {"brooks", "stop", "buy", "sell", "他们", "下一根", "这个", "然后"}
VISUAL_TERMS = ("图中显示", "K线突破", "箭头", "止损线", "跳空", "图表", "画面")


def build_storyline_from_segments(video_id: str, segments: list[TranscriptSegment], query: str | None = None) -> Storyline:
    multimodal = [
        MultimodalSegment(segment_id=segment.segment_id, start=segment.start, end=segment.end, transcript_text=segment.text, evidence_types=["speech"])
        for segment in segments
    ]
    return _fallback_storyline(video_id, multimodal, query)


def build_storyline_from_multimodal_segments(
    video_id: str,
    segments: list[MultimodalSegment],
    query: str | None = None,
    metadata: VideoMetadata | None = None,
    settings: Settings | None = None,
    log_dir: Path | None = None,
) -> Storyline:
    settings = settings or get_settings()
    effective_query = query or DEFAULT_STORYLINE_QUERY
    if settings.has_llm_config and segments:
        try:
            evidence = _retrieve_storyline_evidence(video_id, segments, effective_query, settings)
            client = _storyline_llm_client(settings, log_dir)
            payload = client.generate_json(_storyline_prompt(video_id, metadata, evidence, effective_query), _schema_hint())
            storyline = _coerce_llm_storyline(video_id, segments, effective_query, payload)
            _validate_storyline_nodes(storyline, segments)
            return storyline
        except Exception as exc:  # noqa: BLE001
            if settings.strict_runtime or settings.strict_llm:
                raise RuntimeError(f"LLM storyline failed in strict mode: {exc}") from exc
            return _fallback_storyline(video_id, segments, effective_query)
    return _fallback_storyline(video_id, segments, effective_query)


def _storyline_llm_client(settings: Settings, log_dir: Path | None) -> LLMClient:
    if settings.has_storyline_llm_config:
        return LLMClient(
            settings,
            log_dir,
            api_key=settings.storyline_api_key,
            base_url=settings.runtime_storyline_base_url,
            model=settings.runtime_storyline_model,
            validate_runtime_endpoint=False,
        )
    return LLMClient(settings, log_dir)


def save_storyline(storyline: Storyline, video_dir: Path) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    output = video_dir / "storyline.json"
    output.write_text(storyline.model_dump_json(indent=2), encoding="utf-8")
    return output


def load_storyline(path: Path) -> Storyline:
    return Storyline.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _retrieve_storyline_evidence(video_id: str, segments: list[MultimodalSegment], query: str, settings: Settings) -> list[RetrievedEvidence]:
    store = ChromaMemoryStore(settings)
    if not store.has_video_documents(video_id):
        store.add_multimodal_segments(video_id, segments)
    try:
        evidence = store.search(video_id, query, top_k=settings.storyline_top_k, evidence_types=["speech", "frame_caption"])
    except Exception:  # noqa: BLE001
        evidence = []
    if evidence:
        return _merge_time_coverage_evidence(video_id, segments, evidence, max_items=settings.storyline_top_k)
    rows: list[RetrievedEvidence] = []
    for segment in _select_time_coverage_segments(segments, max_items=settings.storyline_top_k):
        if segment.transcript_text.strip():
            rows.append(
                RetrievedEvidence(
                    evidence_id=f"{video_id}:{segment.segment_id}:speech",
                    video_id=video_id,
                    segment_id=segment.segment_id,
                    evidence_type="speech",
                    text=segment.transcript_text,
                    start=segment.start,
                    end=segment.end,
                    score=0.0,
                )
            )
    return rows


def _merge_time_coverage_evidence(
    video_id: str,
    segments: list[MultimodalSegment],
    evidence: list[RetrievedEvidence],
    max_items: int,
) -> list[RetrievedEvidence]:
    max_items = max(1, max_items)
    merged: list[RetrievedEvidence] = []
    seen: set[str] = set()
    coverage_budget = min(max_items, max(3, max_items // 2))
    for segment in _select_time_coverage_segments(segments, coverage_budget):
        if not segment.transcript_text.strip():
            continue
        evidence_id = f"{video_id}:{segment.segment_id}:speech"
        merged.append(
            RetrievedEvidence(
                evidence_id=evidence_id,
                video_id=video_id,
                segment_id=segment.segment_id,
                evidence_type="speech",
                text=segment.transcript_text,
                start=segment.start,
                end=segment.end,
                score=0.0,
            )
        )
        seen.add(evidence_id)
        seen.add(segment.segment_id)
    for item in evidence:
        if len(merged) >= max_items:
            break
        key = item.evidence_id or item.segment_id
        if key in seen or item.segment_id in seen:
            continue
        merged.append(item)
        seen.add(key)
        seen.add(item.segment_id)
    return sorted(merged[:max_items], key=lambda item: (item.start, item.end))


def _select_time_coverage_segments(segments: list[MultimodalSegment], max_items: int) -> list[MultimodalSegment]:
    usable = [segment for segment in segments if segment.transcript_text.strip() or segment.visual_summary.strip()]
    if len(usable) <= max_items:
        return usable
    if max_items <= 1:
        return [usable[0]]
    indexes = sorted({round(index * (len(usable) - 1) / (max_items - 1)) for index in range(max_items)})
    return [usable[index] for index in indexes]


def _storyline_prompt(video_id: str, metadata: VideoMetadata | None, evidence: list[RetrievedEvidence], query: str) -> str:
    evidence_block = "\n\n".join(_format_evidence(item) for item in evidence[:8])
    title = metadata.title if metadata else video_id
    return f"""
你是 Video Knowledge Agent 的 Query-Guided Storyline 生成器。
只基于 evidence 生成 4 到 7 个语义节点，不要复读 ASR 原文，不要编造。
节点要表达信息推进：问题 -> 机制/原因 -> 风险/方法 -> 结论。
OCR 已禁用，只允许引用 speech、frame、frame_caption。

每个 node 必须包含：
node_id, title, time_start, time_end, summary, key_points, evidence_refs, confidence, status。
summary 用 1 到 2 句中文；key_points 用 2 到 3 条。
status 只能是 supported、weakly_supported、needs_review。
证据不足时设为 needs_review。
只输出 JSON，不要 Markdown。

video_id={video_id}
title={title}
query={query}

evidence:
{evidence_block}
"""


def _schema_hint() -> str:
    return """
{
  "nodes": [
    {
      "node_id": "node_0000",
      "title": "Stop market 的触发与成交机制",
      "time_start": 0.0,
      "time_end": 60.0,
      "summary": "2-4 Chinese sentences.",
      "key_points": ["...", "..."],
      "evidence_refs": [
        {"evidence_id": "video:seg_0000:speech", "segment_id": "seg_0000", "evidence_type": "speech", "timestamp": "[00:00:00]"}
      ],
      "confidence": 0.82,
      "status": "supported"
    }
  ]
}
"""


def _coerce_llm_storyline(video_id: str, segments: list[MultimodalSegment], query: str, payload: dict[str, Any]) -> Storyline:
    valid_segments = {segment.segment_id: segment for segment in segments}
    nodes: list[StorylineNode] = []
    for index, item in enumerate(payload.get("nodes") or []):
        refs = _coerce_refs(item.get("evidence_refs"))
        evidence_ids = _valid_segment_ids([ref.get("segment_id") for ref in refs], valid_segments)
        if not evidence_ids:
            evidence_ids = _valid_segment_ids(item.get("evidence_segment_ids"), valid_segments)
        title = _clean_title(str(item.get("title") or item.get("topic") or f"语义节点 {index + 1}"), index)
        summary = str(item.get("summary") or item.get("claim") or "当前证据不足，需要复核。")
        confidence = max(0.0, min(1.0, float(item.get("confidence") if item.get("confidence") is not None else (0.8 if evidence_ids else 0.3))))
        status = str(item.get("status") or "needs_review")
        if status not in {"supported", "weakly_supported", "needs_review"}:
            status = "needs_review"
        if not evidence_ids or confidence < 0.75:
            status = "weakly_supported" if evidence_ids else "unsupported"
        time_start = float(item.get("time_start") if item.get("time_start") is not None else _segment_start(evidence_ids, valid_segments))
        time_end = float(item.get("time_end") if item.get("time_end") is not None else _segment_end(evidence_ids, valid_segments))
        frame_ids = _frame_ids_for_segments(evidence_ids, valid_segments)
        frame_caption_refs = [ref for ref in (refs or _refs_from_segments(video_id, evidence_ids, valid_segments)) if ref.get("evidence_type") == "frame_caption"]
        modalities = _modalities_for_segments(evidence_ids, valid_segments)
        nodes.append(
            StorylineNode(
                node_id=str(item.get("node_id") or f"node_{index:04d}"),
                time_start=time_start,
                time_end=time_end,
                topic=title,
                claim=summary[:220],
                title=title,
                summary=summary,
                key_points=[str(point) for point in (item.get("key_points") or [])][:5],
                evidence_refs=refs or _refs_from_segments(video_id, evidence_ids, valid_segments),
                evidence_segment_ids=evidence_ids,
                speech_evidence_ids=evidence_ids,
                frame_evidence_ids=frame_ids,
                frame_caption_evidence=frame_caption_refs,
                ocr_evidence=[],
                support_modalities=modalities,
                modality_support={"speech": 1.0 if "speech" in modalities else 0.0, "vision": 1.0 if "frame_caption" in modalities or frame_ids else 0.0},
                confidence=confidence,
                uncertainty=round(1.0 - confidence, 3),
                status=EvidenceStatus(status),
            )
        )
    storyline = Storyline(video_id=video_id, query=query, nodes=nodes or _fallback_storyline(video_id, segments, query).nodes)
    _fill_storyline_metadata(storyline)
    return storyline


def _validate_storyline_nodes(storyline: Storyline, segments: list[MultimodalSegment]) -> None:
    by_id = {segment.segment_id: segment for segment in segments}
    for node in storyline.nodes:
        related = [by_id[item] for item in node.evidence_segment_ids if item in by_id]
        has_frame_caption = any("frame_caption" in segment.evidence_types for segment in related)
        visual_claim = any(term in (node.summary or node.claim) for term in VISUAL_TERMS)
        if not node.evidence_refs and node.status != EvidenceStatus.unsupported:
            node.status = EvidenceStatus.needs_review
        elif node.evidence_refs and not has_frame_caption and (visual_claim or node.confidence < 0.75):
            node.status = EvidenceStatus.weakly_supported
        if node.status == EvidenceStatus.supported and node.confidence < 0.75:
            node.status = EvidenceStatus.weakly_supported
    _fill_storyline_metadata(storyline)


def _fallback_storyline(video_id: str, segments: list[MultimodalSegment], query: str | None) -> Storyline:
    nodes: list[StorylineNode] = []
    for index, segment in enumerate(segments):
        source = " ".join((segment.transcript_text or segment.visual_summary).split())
        evidence_ids = [segment.segment_id] if source else []
        title = _semantic_fallback_title(source, index)
        summary = _semantic_fallback_summary(source)
        nodes.append(
            StorylineNode(
                node_id=f"node_{index:04d}",
                time_start=segment.start,
                time_end=segment.end,
                topic=title,
                claim=summary,
                title=title,
                summary=summary,
                key_points=_fallback_key_points(source),
                evidence_refs=_refs_from_segments(video_id, evidence_ids, {segment.segment_id: segment}),
                evidence_segment_ids=evidence_ids,
                speech_evidence_ids=evidence_ids if segment.transcript_text.strip() else [],
                frame_evidence_ids=list(segment.representative_frame_ids),
                frame_caption_evidence=[
                    {
                        "frame_id": caption.frame_id,
                        "timestamp": format_timestamp(caption.timestamp),
                        "caption": caption.caption,
                    }
                    for caption in segment.visual_captions
                    if caption.caption.strip()
                ],
                ocr_evidence=[],
                support_modalities=[item for item in segment.evidence_types if item in {"speech", "frame", "frame_caption"}],
                modality_support={"speech": 1.0 if segment.transcript_text.strip() else 0.0, "vision": 1.0 if segment.visual_captions or segment.representative_frame_ids else 0.0},
                confidence=0.55 if evidence_ids else 0.2,
                uncertainty=0.45 if evidence_ids else 0.8,
                status=EvidenceStatus.weakly_supported if evidence_ids else EvidenceStatus.needs_review,
            )
        )
    storyline = Storyline(video_id=video_id, query=query or DEFAULT_STORYLINE_QUERY, nodes=nodes)
    _fill_storyline_metadata(storyline)
    return storyline


def _fill_storyline_metadata(storyline: Storyline) -> None:
    warnings: list[str] = []
    speech = frame = frame_caption = 0
    covered_segments: set[str] = set()
    for node in storyline.nodes:
        covered_segments.update(node.evidence_segment_ids)
        speech += len(node.speech_evidence_ids)
        frame += len(node.frame_evidence_ids)
        frame_caption += len(node.frame_caption_evidence)
        if node.status == EvidenceStatus.supported and not node.evidence_segment_ids:
            warnings.append(f"{node.node_id}: supported_without_segment_evidence")
        if not node.frame_caption_evidence:
            warnings.append(f"{node.node_id}: missing_frame_caption_evidence")
    storyline.validation_warnings = warnings
    storyline.evidence_coverage = {"covered_segment_ids": sorted(covered_segments), "node_count": len(storyline.nodes)}
    storyline.modality_usage_summary = {"speech_refs": speech, "frame_refs": frame, "frame_caption_refs": frame_caption}


def _format_evidence(item: RetrievedEvidence) -> str:
    ts = format_timestamp(item.timestamp if item.timestamp is not None else item.start)
    return (
        f"evidence_id={item.evidence_id}\n"
        f"segment_id={item.segment_id}\n"
        f"type={item.evidence_type}\n"
        f"time={ts}\n"
        f"start={item.start:.2f}\n"
        f"end={item.end:.2f}\n"
        f"frame_id={item.frame_id or ''}\n"
        f"text={item.text[:420]}"
    )


def _clean_title(title: str, index: int) -> str:
    title = re.sub(r"\s+", " ", title).strip(" -:：")
    if not title or title.lower() in BAD_TITLE_WORDS or len(title) < 4:
        return f"语义推进节点 {index + 1}"
    return title[:60]


def _semantic_fallback_title(source: str, index: int) -> str:
    if not source:
        return f"证据不足节点 {index + 1}"
    for marker in ("区别", "机制", "风险", "优势", "失效", "方法", "结论", "问题"):
        if marker in source:
            return f"{marker}相关的核心说明"
    return f"语义推进节点 {index + 1}"


def _semantic_fallback_summary(source: str) -> str:
    if not source:
        return "当前片段缺少足够 speech 或 frame_caption 证据，需要复核。"
    return source[:220]


def _fallback_key_points(source: str) -> list[str]:
    if not source:
        return ["缺少可用证据", "需要补充检索或补帧"]
    chunks = re.split(r"[。！？；;]", source)
    return [chunk.strip()[:90] for chunk in chunks if chunk.strip()][:5] or [source[:90]]


def _coerce_refs(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs = []
    for item in value:
        if isinstance(item, dict) and str(item.get("evidence_type", "")) in {"speech", "frame", "frame_caption"}:
            refs.append({key: item.get(key) for key in ("evidence_id", "segment_id", "evidence_type", "timestamp", "frame_id") if item.get(key) is not None})
    return refs


def _refs_from_segments(video_id: str, ids: list[str], segments: dict[str, MultimodalSegment]) -> list[dict[str, Any]]:
    refs = []
    for segment_id in ids:
        segment = segments.get(segment_id)
        if not segment:
            continue
        refs.append(
            {
                "evidence_id": f"{video_id}:{segment_id}:speech",
                "segment_id": segment_id,
                "evidence_type": "speech",
                "timestamp": format_timestamp(segment.start),
            }
        )
        for caption in segment.visual_captions[:3]:
            if caption.caption.strip():
                refs.append(
                    {
                        "evidence_id": f"{video_id}:{segment_id}:frame_caption:{caption.frame_id}",
                        "segment_id": segment_id,
                        "evidence_type": "frame_caption",
                        "timestamp": format_timestamp(caption.timestamp),
                        "frame_id": caption.frame_id,
                    }
                )
    return refs


def _valid_segment_ids(value: object, segments: dict[str, MultimodalSegment]) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        segment_id = str(item)
        if segment_id in segments and segment_id not in result:
            result.append(segment_id)
    return result


def _segment_start(ids: list[str], segments: dict[str, MultimodalSegment]) -> float:
    return min((segments[item].start for item in ids if item in segments), default=0.0)


def _segment_end(ids: list[str], segments: dict[str, MultimodalSegment]) -> float:
    return max((segments[item].end for item in ids if item in segments), default=0.0)


def _frame_ids_for_segments(ids: list[str], segments: dict[str, MultimodalSegment]) -> list[str]:
    result = []
    for segment_id in ids:
        for frame_id in segments.get(segment_id, MultimodalSegment(segment_id="x", start=0, end=0)).representative_frame_ids:
            if frame_id not in result:
                result.append(frame_id)
    return result


def _modalities_for_segments(ids: list[str], segments: dict[str, MultimodalSegment]) -> list[str]:
    result = []
    for segment_id in ids:
        for modality in segments.get(segment_id, MultimodalSegment(segment_id="x", start=0, end=0)).evidence_types:
            if modality not in result:
                result.append(modality)
    return result or ["speech"]
