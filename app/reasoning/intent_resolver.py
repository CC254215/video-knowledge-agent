from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.models import MultimodalSegment, ResolvedIntent, VideoMetadata
from app.modality.router import question_needs_visual_check
from app.reasoning.llm_client import LLMClient


ACTION_HINTS = ("操作建议", "操作上的建议", "实操建议", "行动建议", "风控建议", "交易建议", "建议", "怎么做", "清单")
VISUAL_OPERATION_HINTS = ("操作界面", "按钮", "点击", "画面", "图中", "视频里这一步", "截图", "界面", "演示")


def resolve_intent(
    question: str,
    metadata: VideoMetadata,
    segments: list[MultimodalSegment],
    settings: Settings | None = None,
    log_dir: Path | None = None,
    use_llm: bool = False,
) -> ResolvedIntent:
    resolved = resolve_intent_rules(question, metadata, segments)
    if not use_llm or not settings or not settings.has_llm_config:
        return resolved
    try:
        llm_result = LLMClient(settings, log_dir).generate_json(
            build_intent_prompt(question, metadata, segments, resolved),
            schema_hint="""
{
  "rewritten_question": "...",
  "question_type": "summary|explanation|evidence_check|comparison|time_navigation|critique|action_items|visual_operation",
  "retrieval_queries": ["..."],
  "required_evidence_types": ["speech"],
  "optional_evidence_types": ["frame_caption"],
  "needs_visual_check": false,
  "can_use_general_knowledge": true,
  "reason": "..."
}
""",
        )
    except Exception:
        return resolved
    return merge_llm_intent(question, resolved, llm_result)


def resolve_intent_rules(question: str, metadata: VideoMetadata, segments: list[MultimodalSegment]) -> ResolvedIntent:
    q = question.strip()
    lower = q.lower()
    if _is_visual_operation(q):
        return ResolvedIntent(
            original_question=q,
            rewritten_question=q,
            question_type="visual_operation",
            retrieval_queries=[q, f"{metadata.title} 画面 演示 操作界面 按钮"],
            required_evidence_types=["frame_caption"],
            optional_evidence_types=["speech", "frame"],
            needs_visual_check=True,
            can_use_general_knowledge=False,
            reason="问题明确询问画面、按钮、界面或演示步骤，需要视觉证据。",
        )
    if _is_action_advice(q):
        keywords = extract_video_keywords(metadata, segments)
        rewritten = f"请基于视频中关于{keywords}的内容，整理可执行的操作建议、风险控制建议和行动清单。"
        return ResolvedIntent(
            original_question=q,
            rewritten_question=rewritten,
            question_type="action_items",
            retrieval_queries=[
                rewritten,
                f"{keywords} 操作建议 风险控制 行动清单",
                "视频中有哪些可执行建议",
            ],
            required_evidence_types=["speech"],
            optional_evidence_types=["frame_caption"],
            needs_visual_check=False,
            can_use_general_knowledge=True,
            reason="用户的“操作建议”更可能指实操/行动建议，不是界面操作。",
        )
    if any(token in q for token in ("分钟", "秒", "时间点", "十分钟")) or any(ch in q for ch in (":", "：")):
        return ResolvedIntent(
            original_question=q,
            rewritten_question=q,
            question_type="time_navigation",
            retrieval_queries=[q, f"{metadata.title} 指定时间点 内容 问题"],
            required_evidence_types=["speech"],
            optional_evidence_types=["frame_caption"],
            needs_visual_check=False,
            can_use_general_knowledge=False,
            reason="问题包含明确时间锚点，优先按时间检索视频证据。",
        )
    needs_visual = question_needs_visual_check(q)
    question_type = _question_type(q)
    return ResolvedIntent(
        original_question=q,
        rewritten_question=q,
        question_type=question_type,
        retrieval_queries=[q],
        required_evidence_types=["frame_caption"] if needs_visual else ["speech"],
        optional_evidence_types=["speech", "frame_caption"] if needs_visual else ["frame_caption"],
        needs_visual_check=needs_visual,
        can_use_general_knowledge=question_type in {"explanation", "action_items", "critique"},
        reason="规则解析得到的默认意图。",
    )


def build_intent_prompt(question: str, metadata: VideoMetadata, segments: list[MultimodalSegment], resolved: ResolvedIntent) -> str:
    transcript_preview = "\n".join(segment.transcript_text[:300] for segment in segments[:6] if segment.transcript_text.strip())
    return f"""
你是视频问答系统的 Intent Resolver。请重构用户真实意图，不要回答问题。

视频标题：{metadata.title}
作者：{metadata.author or ""}
转录摘要：
{transcript_preview}

用户问题：{question}

规则基线：
{resolved.model_dump_json(indent=2)}

要求：
1. “操作建议/实操建议/行动建议”通常是 action_items，主要需要 speech。
2. “操作界面/按钮/点击/画面演示”才是 visual_operation，需要 frame_caption。
3. 输出 JSON。
"""


def merge_llm_intent(question: str, fallback: ResolvedIntent, payload: dict[str, object]) -> ResolvedIntent:
    allowed_evidence = {"speech", "frame", "frame_caption"}
    required = [str(item) for item in payload.get("required_evidence_types", fallback.required_evidence_types) if str(item) in allowed_evidence]  # type: ignore[union-attr]
    optional = [str(item) for item in payload.get("optional_evidence_types", fallback.optional_evidence_types) if str(item) in allowed_evidence]  # type: ignore[union-attr]
    retrieval = payload.get("retrieval_queries")
    return ResolvedIntent(
        original_question=question,
        rewritten_question=str(payload.get("rewritten_question") or fallback.rewritten_question),
        question_type=str(payload.get("question_type") or fallback.question_type),
        retrieval_queries=[str(item) for item in retrieval] if isinstance(retrieval, list) and retrieval else fallback.retrieval_queries,
        required_evidence_types=required or fallback.required_evidence_types,  # type: ignore[arg-type]
        optional_evidence_types=optional or fallback.optional_evidence_types,  # type: ignore[arg-type]
        needs_visual_check=bool(payload.get("needs_visual_check", fallback.needs_visual_check)),
        can_use_general_knowledge=bool(payload.get("can_use_general_knowledge", fallback.can_use_general_knowledge)),
        general_knowledge_policy=fallback.general_knowledge_policy,
        reason=str(payload.get("reason") or fallback.reason),
    )


def extract_video_keywords(metadata: VideoMetadata, segments: list[MultimodalSegment]) -> str:
    text = f"{metadata.title} " + " ".join(segment.transcript_text[:160] for segment in segments[:5])
    candidates = []
    for token in ("交易", "风险控制", "仓位", "极端行情", "交易曲线", "气球", "物理", "Agent", "记忆", "视频"):
        if token.lower() in text.lower():
            candidates.append(token)
    return "、".join(candidates[:6]) or metadata.title


def _is_action_advice(question: str) -> bool:
    return any(token in question for token in ACTION_HINTS) and not _is_visual_operation(question)


def _is_visual_operation(question: str) -> bool:
    return any(token in question for token in VISUAL_OPERATION_HINTS)


def _question_type(question: str) -> str:
    if any(token in question for token in ("总结", "核心", "概括", "主要论点")):
        return "summary"
    if any(token in question for token in ("证据", "依据", "是否支持", "证明")):
        return "evidence_check"
    if any(token in question for token in ("对比", "比较", "区别")):
        return "comparison"
    if any(token in question for token in ("漏洞", "反驳", "批判", "局限")):
        return "critique"
    if any(token in question for token in ("为什么", "解释", "什么意思", "怎么理解")):
        return "explanation"
    return "explanation"
