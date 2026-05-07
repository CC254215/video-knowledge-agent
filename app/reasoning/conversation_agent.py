from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.evidence.orchestrator import EvidenceOrchestrator
from app.evidence.trace import EvidenceTraceRecorder, evidence_rows_for_trace
from app.memory.mempalace_adapter import build_memory_context
from app.memory.promotion import promote_conversation_turn
from app.models import ConfidenceLevel, ConversationTurn, MultimodalSegment, ResolvedIntent, RetrievedEvidence, VideoMetadata
from app.modality.router import compute_segment_modality_weight, question_needs_visual_check
from app.reasoning.evidence_gap_agent import extract_time_anchors
from app.reasoning.intent_resolver import resolve_intent
from app.reasoning.llm_client import LLMClient
from app.reasoning.summarizer import format_timestamp
from app.retrieval.chroma_memory import ChromaMemoryStore
from app.retrieval.dpp import greedy_multimodal_dpp_select
from app.retrieval.embeddings import HashingFallbackEmbedder
from app.retrieval.evidence import compute_retrieved_evidence_agreement, confidence_from_agreement
from app.retrieval.hybrid_retriever import HybridEvidenceRetriever
from app.transcript.multimodal_segmenter import load_multimodal_segments
from app.vision.frame_refiner import FrameEvidenceRefiner, should_refine_frames

INSUFFICIENT_EVIDENCE = "当前视频证据不足以支持这个结论。"


class ConversationAgent:
    def __init__(
        self,
        video_id: str,
        video_dir: str | Path,
        settings: Settings | None = None,
        store: ChromaMemoryStore | None = None,
    ) -> None:
        self.video_id = video_id
        self.video_dir = Path(video_dir)
        self.settings = settings or get_settings()
        self.store = store or ChromaMemoryStore(self.settings)
        self.retriever = HybridEvidenceRetriever(self.store, self.settings)
        self.embedder = HashingFallbackEmbedder()
        self._active_trace: EvidenceTraceRecorder | None = None

    def answer(self, user_question: str, conversation_id: str | None = None) -> ConversationTurn:
        trace = EvidenceTraceRecorder(self.video_id, self.video_dir, user_question)
        self._active_trace = trace
        trace.record("question_received", {"conversation_id": conversation_id})
        segments = self._load_segments()
        if not segments:
            turn = self._insufficient_turn(user_question, "没有可用的 multimodal_segments.json。")
            return self._finalize_turn(turn, trace)
        trace.record("segments_loaded", {"count": len(segments)})

        for segment in segments:
            segment.modality_weight = compute_segment_modality_weight(segment, user_question)
        self.store.add_multimodal_segments(self.video_id, segments)

        turn = self._answer_once(user_question, segments, refined=False)
        trace.record("initial_turn", _turn_trace_payload(turn))
        orchestration = EvidenceOrchestrator(self.video_id, self.video_dir, self.settings, self.store, trace=trace).review_and_maybe_refine(
            user_question,
            turn,
            segments,
            _retrieved_from_bundle(self.video_id, turn.evidence),
            self._load_metadata,
            lambda updated_segments, refined: self._answer_once(user_question, updated_segments, refined=refined),
        )
        turn = orchestration.turn
        segments = orchestration.segments
        if orchestration.handled:
            return self._finalize_turn(turn, trace)

        if self._should_auto_refine(user_question, turn, segments):
            ranges = _target_ranges(turn.evidence, segments)
            trace.record("legacy_auto_refine_started", {"ranges": [{"start": start, "end": end} for start, end in ranges[:2]]})
            reason_parts: list[str] = []
            for start, end in ranges[:2]:
                segments = FrameEvidenceRefiner(self.video_dir, settings=self.settings).refine(
                    self.video_id,
                    user_question,
                    (start, end),
                    segments,
                )
                reason_parts.append(f"{format_timestamp(start)}-{format_timestamp(end)}")
            self.store.add_multimodal_segments(self.video_id, segments)
            turn = self._answer_once(user_question, segments, refined=True)
            turn.reason = f"初始证据不足，已对 {', '.join(reason_parts)} 时间段补充抽帧后重新回答。{turn.reason}"
            trace.record("legacy_auto_refine_completed", _turn_trace_payload(turn))

        return self._finalize_turn(turn, trace)

    def _answer_once(self, user_question: str, segments: list[MultimodalSegment], refined: bool) -> ConversationTurn:
        resolved_intent = resolve_intent(user_question, self._load_metadata(), segments, settings=self.settings, log_dir=self.video_dir / "llm_logs", use_llm=False)
        question_type = resolved_intent.question_type
        needs_visual = resolved_intent.needs_visual_check
        if self._active_trace:
            self._active_trace.record("intent_resolved", resolved_intent.model_dump(mode="json"))
        query_candidates = self._search_for_intent(resolved_intent, top_k=12, segments=segments)
        query_evidence = self._select_evidence(query_candidates, k=5)
        if self._active_trace:
            self._active_trace.record(
                "query_evidence_selected",
                {
                    "candidates": evidence_rows_for_trace(query_candidates),
                    "selected": evidence_rows_for_trace(query_evidence),
                },
            )
        if not query_evidence:
            return self._insufficient_turn(user_question, "未检索到相关 speech/frame_caption 证据。")

        llm_result = self._generate_llm_answer(user_question, question_type, query_evidence, needs_visual, resolved_intent)
        candidate_answer = str(llm_result.get("answer") or compose_grounded_answer(question_type, query_evidence))
        answer_candidates = self.retriever.search(
            self.video_id,
            candidate_answer,
            segments,
            top_k=12,
            evidence_types=["speech", "frame_caption"],
        )
        answer_evidence = self._select_evidence(answer_candidates, k=5)
        if self._active_trace:
            self._active_trace.record(
                "answer_evidence_selected",
                {
                    "candidates": evidence_rows_for_trace(answer_candidates),
                    "selected": evidence_rows_for_trace(answer_evidence),
                },
            )
        agreement = compute_retrieved_evidence_agreement(query_evidence, answer_evidence)
        confidence = _merge_confidence(llm_result.get("confidence"), confidence_from_agreement(agreement))

        cited_ids = _valid_evidence_ids(llm_result.get("evidence_ids"), query_evidence)
        selected = [item for item in query_evidence if item.evidence_id in cited_ids or item.segment_id in cited_ids] if cited_ids else query_evidence
        requires_frame_caption = "frame_caption" in resolved_intent.required_evidence_types
        if requires_frame_caption and not any(item.evidence_type == "frame_caption" and item.text.strip() for item in selected):
            confidence = ConfidenceLevel.low
            if refined:
                candidate_answer = f"{INSUFFICIENT_EVIDENCE} 已检查相关时间段，但缺少可用 frame_caption 视觉描述。"

        if INSUFFICIENT_EVIDENCE in candidate_answer:
            confidence = ConfidenceLevel.low

        evidence_bundle = build_evidence_bundle(selected, segments)
        evidence_types = sorted({item.get("evidence_type") for item in evidence_bundle if item.get("evidence_type") in {"speech", "frame", "frame_caption"}})
        return ConversationTurn(
            turn_id=str(uuid.uuid4()),
            video_id=self.video_id,
            user_question=user_question,
            answer=candidate_answer if selected else INSUFFICIENT_EVIDENCE,
            evidence_ids=_unique([item.segment_id for item in selected]),
            evidence=evidence_bundle,
            timestamps=[item["timestamp"] for item in evidence_bundle],
            evidence_types=evidence_types,  # type: ignore[arg-type]
            confidence=confidence,
            agreement_score=round(agreement, 4),
            query_evidence_ids=[item.evidence_id for item in query_evidence],
            answer_evidence_ids=[item.evidence_id for item in answer_evidence],
            dpp_used=True,
            dpp_details={
                "query_selected_evidence_ids": [item.evidence_id for item in query_evidence],
                "answer_selected_evidence_ids": [item.evidence_id for item in answer_evidence],
                "query_evidence_types": sorted({item.evidence_type for item in query_evidence}),
                "answer_evidence_types": sorted({item.evidence_type for item in answer_evidence}),
            },
            needs_visual_check=needs_visual,
            reason=(
                f"question_type={question_type}; intent_rewritten={resolved_intent.rewritten_question}; intent_reason={resolved_intent.reason}; agreement={agreement:.2f}; "
                f"query_evidence={len(query_evidence)}; answer_evidence={len(answer_evidence)}; "
                f"mempalace_context_items={llm_result.get('memory_context_items', 0)}; "
                f"llm_reason={llm_result.get('reason', '')}"
            ),
            suggested_followup_questions=_coerce_followups(llm_result.get("suggested_followup_questions"), question_type, needs_visual),
        )

    def _select_evidence(self, candidates: list[RetrievedEvidence], k: int) -> list[RetrievedEvidence]:
        if not candidates:
            return []
        vectors = self.embedder.embed_texts([item.text for item in candidates])
        selected_ids = greedy_multimodal_dpp_select(
            [item.evidence_id for item in candidates],
            [item.score for item in candidates],
            vectors,
            [item.evidence_type for item in candidates],
            k=min(k, len(candidates)),
            min_relevance=0.0,
        )
        order = {evidence_id: index for index, evidence_id in enumerate(selected_ids)}
        return sorted([item for item in candidates if item.evidence_id in order], key=lambda item: order[item.evidence_id])

    def _search_for_intent(self, resolved_intent: ResolvedIntent, top_k: int, segments: list[MultimodalSegment] | None = None) -> list[RetrievedEvidence]:
        evidence_types = _intent_evidence_types(resolved_intent)
        if resolved_intent.question_type == "time_navigation" and segments:
            time_results = _time_filtered_evidence(self.video_id, resolved_intent.original_question, segments, evidence_types)
            if time_results:
                if self._active_trace:
                    self._active_trace.record("time_filtered_retrieval", {"results": evidence_rows_for_trace(time_results)})
                return time_results[:top_k]
        merged: dict[str, RetrievedEvidence] = {}
        for query in resolved_intent.retrieval_queries or [resolved_intent.rewritten_question]:
            results = self.retriever.search(self.video_id, query, segments or [], top_k=top_k, evidence_types=evidence_types)
            if self._active_trace:
                self._active_trace.record("retrieval_query", self.retriever.last_trace)
            for item in results:
                existing = merged.get(item.evidence_id)
                if existing is None or item.score > existing.score:
                    merged[item.evidence_id] = item
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def _finalize_turn(self, turn: ConversationTurn, trace: EvidenceTraceRecorder) -> ConversationTurn:
        turn.reason = f"trace_id={trace.trace_id}; {turn.reason}"
        trace.record("final_turn", _turn_trace_payload(turn))
        promotion = promote_conversation_turn(turn, self._load_metadata(), self.settings, trace_id=trace.trace_id)
        trace.record("memory_promotion", promotion)
        trace.write()
        self._append_turn(turn)
        self._active_trace = None
        return turn

    def _should_auto_refine(self, question: str, turn: ConversationTurn, segments: list[MultimodalSegment]) -> bool:
        if not should_refine_frames(
            question,
            confidence=turn.confidence.value,
            agreement=turn.agreement_score,
            transcript_text=_related_transcript(turn.evidence, segments),
        ):
            return False
        if turn.agreement_score >= 0.3 and turn.confidence != ConfidenceLevel.low:
            return False
        return question_needs_visual_check(question) or _transcript_has_visual_trigger(_related_transcript(turn.evidence, segments))

    def _load_segments(self) -> list[MultimodalSegment]:
        path = self.video_dir / "multimodal_segments.json"
        return load_multimodal_segments(path) if path.exists() else []

    def _load_metadata(self) -> VideoMetadata:
        path = self.video_dir / "metadata.json"
        if path.exists():
            return VideoMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return VideoMetadata(video_id=self.video_id, title=self.video_id)

    def _load_history(self, limit: int = 6) -> list[dict[str, Any]]:
        path = self.video_dir / "qa_history.json"
        if not path.exists():
            return []
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return history[-limit:]

    def _append_turn(self, turn: ConversationTurn) -> None:
        path = self.video_dir / "qa_history.json"
        history = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                history = []
        history.append(turn.model_dump(mode="json"))
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.video_dir / "qa_history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(turn.model_dump_json() + "\n")

    def _insufficient_turn(self, question: str, reason: str) -> ConversationTurn:
        return ConversationTurn(
            turn_id=str(uuid.uuid4()),
            video_id=self.video_id,
            user_question=question,
            answer=INSUFFICIENT_EVIDENCE,
            evidence_ids=[],
            evidence=[],
            timestamps=[],
            evidence_types=[],
            confidence=ConfidenceLevel.low,
            agreement_score=0.0,
            query_evidence_ids=[],
            answer_evidence_ids=[],
            dpp_used=False,
            dpp_details={},
            needs_visual_check=question_needs_visual_check(question),
            reason=reason,
            suggested_followup_questions=["是否要手动指定时间段补帧？", "是否换一个更具体的问题重新检索？"],
        )

    def _generate_llm_answer(
        self,
        question: str,
        question_type: str,
        evidence: list[RetrievedEvidence],
        needs_visual: bool,
        resolved_intent: ResolvedIntent | None = None,
    ) -> dict[str, Any]:
        if not self.settings.has_llm_config:
            if needs_visual:
                return {
                    "answer": f"{INSUFFICIENT_EVIDENCE} 该问题需要可靠的 frame_caption 视觉理解，但当前 LLM 不可用，不能只凭检索片段推断画面细节。",
                    "evidence_ids": [item.evidence_id for item in evidence],
                    "confidence": "low",
                    "reason": "视觉问题在 LLM 不可用时不使用本地拼接回答。",
                    "suggested_followup_questions": suggest_followups(question_type, needs_visual),
                }
            return {
                "answer": compose_grounded_answer(question_type, evidence),
                "evidence_ids": [item.evidence_id for item in evidence],
                "confidence": "medium",
                "reason": "LLM 未配置，使用本地证据拼接回答。",
                "suggested_followup_questions": suggest_followups(question_type, needs_visual),
            }
        memory_context = build_memory_context(question, limit=5)
        prompt = build_conversation_prompt(question, question_type, evidence, self._load_history(), needs_visual, memory_context, resolved_intent)
        schema_hint = """
{
  "answer": "Chinese synthesized answer. Lead with the conclusion grounded in video evidence, then cite key evidence. If useful, add clearly labeled long-term memory/background knowledge. If evidence is insufficient, answer 当前视频证据不足以支持这个结论。",
  "evidence_ids": ["video:seg_0000:speech"],
  "confidence": "high|medium|low",
  "reason": "short evidence quality reason",
  "suggested_followup_questions": ["..."]
}
"""
        try:
            result = LLMClient(self.settings, self.video_dir / "llm_logs").generate_json(prompt, schema_hint=schema_hint)
        except Exception as exc:  # noqa: BLE001
            if needs_visual:
                return {
                    "answer": f"{INSUFFICIENT_EVIDENCE} 该问题需要可靠的 frame_caption 视觉理解，但 LLM 调用失败，不能只凭检索片段推断画面细节。",
                    "evidence_ids": [item.evidence_id for item in evidence],
                    "confidence": "low",
                    "reason": f"视觉问题 LLM 调用失败，拒绝本地拼接回答：{exc}; mempalace_context_items={len(memory_context.get('long_term_memory', []))}",
                    "suggested_followup_questions": suggest_followups(question_type, needs_visual),
                }
            return {
                "answer": compose_grounded_answer(question_type, evidence),
                "evidence_ids": [item.evidence_id for item in evidence],
                "confidence": "medium",
                "reason": f"LLM 调用失败，使用本地规则回答：{exc}; mempalace_context_items={len(memory_context.get('long_term_memory', []))}",
                "suggested_followup_questions": suggest_followups(question_type, needs_visual),
            }
        if isinstance(result, dict):
            result["memory_context_items"] = len(memory_context.get("long_term_memory", []))
            return result
        return {}


def build_conversation_prompt(
    question: str,
    question_type: str,
    evidence: list[RetrievedEvidence],
    history: list[dict[str, Any]],
    needs_visual: bool,
    memory_context: dict[str, Any] | None = None,
    resolved_intent: ResolvedIntent | None = None,
) -> str:
    evidence_block = "\n".join(_format_evidence_for_prompt(item) for item in evidence)
    history_block = "\n".join(
        f"- Q: {item.get('user_question', '')}\n  A: {item.get('answer', '')}\n  Evidence: {item.get('evidence_ids', [])}"
        for item in history
    ) or "无"
    memory_block = json.dumps(memory_context or {"long_term_memory": [], "answer_policy": {}}, ensure_ascii=False, indent=2)
    intent_block = resolved_intent.model_dump_json(indent=2) if resolved_intent else "{}"
    return f"""
你是 Video Knowledge Agent。你的回答应该像一个综合分析助理，而不是证据摘录器。
当前阶段 OCR 已禁用。唯一允许的 evidence_types 是 speech、frame、frame_caption。
规则：
1. 视频事实和“视频里说了什么”的结论，必须主要基于 video_evidence；不要编造视频中没有出现的内容。
2. 每个实质视频结论必须引用 evidence_ids 中的 evidence_id。
3. 如果证据不足，answer 必须写：{INSUFFICIENT_EVIDENCE}
4. 如果问题涉及画面/图表/按钮/界面/公式/代码，但没有 frame_caption，必须判断证据不足。
5. 不要引用 OCR，不要输出 ocr evidence_type。
6. 输出必须是 JSON。
7. 不要把 evidence 文本原样堆出来。先用自己的话回答问题，再用 2-4 个关键依据支撑结论。
8. MemPalace long_term_memory_context 只能作为长期背景、用户偏好或历史洞察，不能替代当前视频证据。
9. 如果使用长期记忆，必须明确标注为“长期记忆补充”；所有“当前视频说了什么”的事实结论仍必须引用 video_evidence。
10. 可以使用少量通用背景知识帮助解释、归纳或指出常识性含义，但必须明确标注为“背景知识补充”，并且不能覆盖或反驳当前视频证据。
11. 推荐 answer 结构：直接结论 -> 视频证据依据 -> 长期记忆补充/背景知识补充（可选） -> 限制或不确定性。
12. 长期记忆和背景知识的使用策略是 only_after_video_evidence_and_clearly_labeled。

question_type={question_type}
needs_visual={needs_visual}

resolved_intent:
{intent_block}

recent_history:
{history_block}

long_term_memory_context:
{memory_block}

user_question:
{question}

video_evidence:
{evidence_block}
"""

def _format_evidence_for_prompt(item: RetrievedEvidence) -> str:
    timestamp = item.timestamp if item.timestamp is not None else item.start
    return (
        f"evidence_id={item.evidence_id}\n"
        f"segment_id={item.segment_id}\n"
        f"evidence_type={item.evidence_type}\n"
        f"time={format_timestamp(timestamp)}\n"
        f"score={item.score:.3f}\n"
        f"text={item.text[:900]}\n"
    )


def _retrieved_from_bundle(video_id: str, evidence: list[dict[str, Any]]) -> list[RetrievedEvidence]:
    rows: list[RetrievedEvidence] = []
    for item in evidence:
        evidence_type = item.get("evidence_type")
        if evidence_type not in {"speech", "frame", "frame_caption"}:
            continue
        rows.append(
            RetrievedEvidence(
                evidence_id=str(item.get("evidence_id") or f"{video_id}:{item.get('segment_id', 'unknown')}:{evidence_type}"),
                video_id=video_id,
                segment_id=str(item.get("segment_id") or "unknown"),
                evidence_type=evidence_type,
                text=str(item.get("frame_caption") or item.get("text") or ""),
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or item.get("start") or 0.0),
                frame_id=item.get("frame_id"),
                image_path=item.get("image_path"),
                timestamp=float(item["start"]) if item.get("start") is not None else None,
                score=float(item.get("score") or 0.0),
            )
        )
    return rows


def _intent_evidence_types(intent: ResolvedIntent) -> list[str]:
    ordered: list[str] = []
    for item in [*intent.required_evidence_types, *intent.optional_evidence_types]:
        if item in {"speech", "frame_caption", "frame"} and item not in ordered:
            ordered.append(item)
    return ordered or ["speech", "frame_caption"]


def _time_filtered_evidence(
    video_id: str,
    question: str,
    segments: list[MultimodalSegment],
    evidence_types: list[str],
) -> list[RetrievedEvidence]:
    metadata_duration = max((segment.end for segment in segments), default=None)
    anchors = extract_time_anchors(question, metadata_duration)
    if not anchors:
        return []
    rows: list[RetrievedEvidence] = []
    for anchor in anchors:
        for segment in segments:
            distance = _distance_to_range(anchor.timestamp, segment.start, segment.end)
            if distance > 90:
                continue
            if "speech" in evidence_types and segment.transcript_text.strip():
                rows.append(
                    RetrievedEvidence(
                        evidence_id=f"{video_id}:{segment.segment_id}:speech",
                        video_id=video_id,
                        segment_id=segment.segment_id,
                        evidence_type="speech",
                        text=segment.transcript_text,
                        start=segment.start,
                        end=segment.end,
                        timestamp=max(segment.start, min(anchor.timestamp, segment.end)),
                        score=max(0.0, 1.0 - distance / 90.0),
                    )
                )
            if "frame_caption" in evidence_types:
                for caption in segment.visual_captions:
                    if not caption.caption.strip():
                        continue
                    caption_distance = abs(caption.timestamp - anchor.timestamp)
                    if caption_distance > 90:
                        continue
                    rows.append(
                        RetrievedEvidence(
                            evidence_id=f"{video_id}:{segment.segment_id}:frame_caption:{caption.frame_id}",
                            video_id=video_id,
                            segment_id=segment.segment_id,
                            evidence_type="frame_caption",
                            text=caption.caption,
                            start=segment.start,
                            end=segment.end,
                            frame_id=caption.frame_id,
                            image_path=caption.image_path,
                            timestamp=caption.timestamp,
                            score=max(0.0, 1.0 - caption_distance / 90.0),
                        )
                    )
    return sorted(rows, key=lambda item: (-(item.score), item.start, item.evidence_type))


def _distance_to_range(timestamp: float, start: float, end: float) -> float:
    if start <= timestamp <= end:
        return 0.0
    return min(abs(timestamp - start), abs(timestamp - end))


def build_evidence_bundle(selected: list[RetrievedEvidence], segments: list[MultimodalSegment]) -> list[dict[str, Any]]:
    by_id = {segment.segment_id: segment for segment in segments}
    bundle: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in sorted(selected, key=lambda evidence: (evidence.start, evidence.timestamp or evidence.start, evidence.evidence_type)):
        segment = by_id.get(item.segment_id)
        if not segment:
            continue
        frame_id = item.frame_id
        image_path = item.image_path or _image_path_for_frame(segment, frame_id)
        key = (item.segment_id, item.evidence_type, frame_id)
        if key in seen:
            continue
        seen.add(key)
        bundle.append(
            {
                "evidence_id": item.evidence_id,
                "segment_id": item.segment_id,
                "evidence_type": item.evidence_type,
                "timestamp": format_timestamp(item.timestamp if item.timestamp is not None else item.start),
                "time_start": format_timestamp(item.start),
                "time_end": format_timestamp(item.end),
                "start": item.start,
                "end": item.end,
                "text": _speech_text_for(item, segment),
                "frame_id": frame_id,
                "image_path": image_path,
                "frame_caption": item.text if item.evidence_type == "frame_caption" else _caption_for_frame(segment, frame_id),
                "score": item.score,
            }
        )
    return bundle


def _speech_text_for(item: RetrievedEvidence, segment: MultimodalSegment) -> str:
    if item.evidence_type == "speech":
        return item.text
    return segment.transcript_text


def _caption_for_frame(segment: MultimodalSegment, frame_id: str | None) -> str:
    for caption in segment.visual_captions:
        if frame_id is None or caption.frame_id == frame_id:
            return caption.caption
    return ""


def _image_path_for_frame(segment: MultimodalSegment, frame_id: str | None) -> str | None:
    for caption in segment.visual_captions:
        if frame_id is None or caption.frame_id == frame_id:
            return caption.image_path
    return None


def classify_question(question: str) -> str:
    if any(token in question for token in ("分钟", "秒", "时间点", "十分钟")) or any(ch in question for ch in (":", "：")):
        return "time_navigation"
    if any(token in question for token in ("操作建议", "操作上的建议", "实操建议", "行动建议", "风控建议", "交易建议", "清单", "怎么做")):
        return "action_items"
    if any(token in question for token in ("总结", "核心", "概括", "主要论点")):
        return "summary"
    if any(token in question for token in ("为什么", "解释", "什么意思", "怎么理解")):
        return "explanation"
    if any(token in question for token in ("证据", "依据", "是否支持", "证明")):
        return "evidence_check"
    if any(token in question for token in ("对比", "比较", "区别")):
        return "comparison"
    if any(token in question for token in ("跳到", "时间", "哪里", "哪一段")):
        return "navigation"
    if any(token in question for token in ("漏洞", "反驳", "问题", "批判", "局限")):
        return "critique"
    if any(token in question for token in ("行动", "步骤")):
        return "action_items"
    return "explanation"


def compose_grounded_answer(question_type: str, evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return INSUFFICIENT_EVIDENCE
    snippets = [_clean_evidence_snippet(item.text, 120) for item in evidence if item.text.strip()]
    if not snippets:
        return INSUFFICIENT_EVIDENCE
    lead = {
        "summary": "基于当前视频证据，这段内容的核心可以概括为",
        "explanation": "根据相关片段，可以理解为",
        "evidence_check": "当前视频能支持的判断是",
        "comparison": "视频中可支持的对比结论是",
        "navigation": "相关内容主要出现在这些时间段",
        "time_navigation": "指定时间点附近主要在说明",
        "critique": "基于视频证据，较稳妥的局限判断是",
        "action_items": "视频中能提炼出的可执行建议是",
    }.get(question_type, "根据视频证据，可以回答为")
    synthesized = "；".join(snippets[:3])
    citations = "；".join(
        f"{format_timestamp(item.timestamp if item.timestamp is not None else item.start)} {item.evidence_id}"
        for item in evidence[:4]
    )
    caveat = " 这个回答只使用已检索到的视频证据；如果问题需要画面细节但缺少 frame_caption，需要补充关键帧后再判断。"
    return f"{lead}：{synthesized}。关键依据：{citations}。{caveat}"


def _clean_evidence_snippet(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def suggest_followups(question_type: str, needs_visual: bool) -> list[str]:
    followups = ["这个结论有哪些限制或反例？"]
    if question_type != "evidence_check":
        followups.append("支持这个回答的关键证据在哪里？")
    if needs_visual:
        followups.append("是否需要对相关时间段补充更多关键帧视觉描述？")
    return followups


def _coerce_followups(value: object, question_type: str, needs_visual: bool) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return cleaned[:5]
    return suggest_followups(question_type, needs_visual)


def _valid_evidence_ids(value: object, allowed: list[RetrievedEvidence]) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed_ids = {item.evidence_id for item in allowed} | {item.segment_id for item in allowed}
    result: list[str] = []
    for item in value:
        evidence_id = str(item)
        if evidence_id in allowed_ids and evidence_id not in result:
            result.append(evidence_id)
    return result


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _merge_confidence(llm_confidence: object, agreement_confidence: str) -> ConfidenceLevel:
    levels = {"low": 0, "medium": 1, "high": 2}
    llm_level = levels.get(str(llm_confidence or "medium").lower(), 1)
    agreement_level = levels.get(agreement_confidence, 0)
    merged = min(llm_level, agreement_level)
    return ConfidenceLevel.low if merged == 0 else ConfidenceLevel.medium if merged == 1 else ConfidenceLevel.high


def _turn_trace_payload(turn: ConversationTurn) -> dict[str, Any]:
    return {
        "turn_id": turn.turn_id,
        "answer": turn.answer[:1000],
        "confidence": turn.confidence.value,
        "agreement_score": turn.agreement_score,
        "evidence_ids": turn.evidence_ids,
        "query_evidence_ids": turn.query_evidence_ids,
        "answer_evidence_ids": turn.answer_evidence_ids,
        "evidence_types": turn.evidence_types,
        "needs_visual_check": turn.needs_visual_check,
        "reason": turn.reason,
    }


def _target_ranges(evidence: list[dict[str, Any]], segments: list[MultimodalSegment]) -> list[tuple[float, float]]:
    if evidence:
        ranges = [(float(item.get("start") or 0.0), float(item.get("end") or item.get("start") or 0.0)) for item in evidence]
    else:
        ranges = [(segment.start, segment.end) for segment in segments[:1]]
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 2:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged or [(0.0, 10.0)]


def _related_transcript(evidence: list[dict[str, Any]], segments: list[MultimodalSegment]) -> str:
    segment_ids = {str(item.get("segment_id")) for item in evidence if item.get("segment_id")}
    return " ".join(segment.transcript_text for segment in segments if segment.segment_id in segment_ids)


def _transcript_has_visual_trigger(text: str) -> bool:
    return any(token in text for token in ("如图所示", "这里可以看到", "这个图", "这个画面", "这一步", "这个按钮"))

