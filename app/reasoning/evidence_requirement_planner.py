from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.models import ClaimRequirement, DeterministicQuestionContext, EvidencePlan, QuestionAnalysis, RequiredFact, ScopeRequirement
from app.reasoning.llm_client import LLMClient


def plan_evidence(
    question: str,
    analysis: QuestionAnalysis,
    context: DeterministicQuestionContext,
    settings: Settings | None = None,
    log_dir: Path | None = None,
    use_llm: bool = True,
) -> EvidencePlan:
    fallback = fallback_evidence_plan(analysis, context)
    if not use_llm or not settings or not settings.has_llm_config:
        return fallback
    try:
        payload = LLMClient(settings, log_dir).generate_json(
            build_evidence_planner_prompt(question, analysis, context), schema_hint=evidence_plan_schema()
        )
        plan = EvidencePlan.model_validate({**payload, "planning_method": "llm"})
    except Exception:
        return fallback
    return _ensure_output_language(_apply_constraints(plan, analysis, context), question)


def _ensure_output_language(plan: EvidencePlan, question: str) -> EvidencePlan:
    """Prevent English LLM fact rewrites from biasing Chinese retrieval."""
    if not any("\u4e00" <= char <= "\u9fff" for char in question):
        return plan
    has_cjk = lambda value: any("\u4e00" <= char <= "\u9fff" for char in value)
    required = [
        fact if has_cjk(fact.fact) else fact.model_copy(update={"fact": f"从视频中确认：{question.strip()[:300]}"})
        for fact in plan.required_facts
    ]
    decisions = [item if has_cjk(item) else f"确认视频中与该问题直接相关的内容：{question.strip()[:300]}" for item in plan.decision_facts]
    requirements = [item if has_cjk(item) else "核对视频中的中文语音与相关画面证据" for item in plan.human_readable_requirements]
    reason = plan.reason if has_cjk(plan.reason) else "证据计划已按中文问题生成，并保留必要的证据模态约束。"
    return plan.model_copy(update={"required_facts": required, "decision_facts": decisions, "human_readable_requirements": requirements, "reason": reason})


def build_evidence_planner_prompt(question: str, analysis: QuestionAnalysis, context: DeterministicQuestionContext) -> str:
    planner_input = {
        "question": question,
        "question_analysis": analysis.model_dump(mode="json"),
        "video_modalities": {"has_real_speech": context.has_real_speech, "has_audio": context.has_audio, "visual_available": True},
        "choices": context.choices,
        "explicit_timestamps": [item.model_dump() for item in context.explicit_timestamps],
        "explicit_intervals": [item.model_dump() for item in context.explicit_intervals],
    }
    return f"""
你负责确定可靠回答当前视频问题之前必须观察到哪些事实。不要回答问题，不要生成检索查询，也不要决定抽帧数量或执行补帧。

请生成 required_facts、decision_facts、required_modalities 和结构化证据约束。
所有 required_facts 必须是能够从视频中验证的具体事实，不能写“需要更多信息”或“需要更多证据”。
时间顺序需要至少两个能够定位到不同时间的有效事件证据；证据模态必须能支持所问事实，可以是语音或视觉。
当 temporal_relation 是 stage 时，需要证明各阶段及其顺序。
decision_facts 表示一旦确定就能决定答案的事实，适用于选择题和开放问题。
required_modalities 只能使用 speech、visual。不要虚构当前系统未提供的 OCR 模态。
如果 QuestionAnalysis 已给出 content_modalities，应以它为边界；如果为空，选择能够证明 required_facts 的最小模态集合，不要因为某模态可用就自动把它列为必需。
所有 required_facts、decision_facts、human_readable_requirements 和 reason 必须使用简体中文；不要生成英文事实描述或英文检索词。
请优先输出 claims、relations、scope。scope 只描述回答范围和完整性，不用于否定单条 claim；“哪些项目”默认 multi_event + best_effort，“完整列出所有项目、一个都不要漏”才使用 whole_video + strict。

输入：
{json.dumps(planner_input, ensure_ascii=False, indent=2)}
"""


def evidence_plan_schema() -> str:
    return """
{
  "claims": [{"id":"c1", "description":"可由视频验证的主张", "required": true, "support_requirement":"direct|derived|either", "preferred_modalities":["speech"], "importance":"core|supporting|optional"}],
  "relations": [{"id":"r1", "subject_claim_id":"c1", "object_claim_id":"c2", "description":"结构化关系", "relation_type":"before|after|causes|explains|compares|differs_from|part_of|same_as|other", "required": true}],
  "scope": {"target":"local|relevant_evidence|multi_event|whole_video", "completeness":"none|best_effort|strict", "allow_partial_answer":true, "description":"范围说明"},
  "required_modalities": ["speech", "visual"],
  "required_facts": [{"fact": "observable fact", "modality": "speech|visual"}],
  "decision_facts": ["fact that determines the answer"],
  "min_distinct_timestamps": 1,
  "requires_temporal_order": false,
  "requires_global_coverage": false,
  "requires_visual_identity": false,
  "human_readable_requirements": ["debug description"],
  "reason": "short planning reason"
}
"""


def fallback_evidence_plan(analysis: QuestionAnalysis, context: DeterministicQuestionContext) -> EvidencePlan:
    modalities = list(analysis.content_modalities) or (["speech"] if context.has_real_speech else ["visual"])
    if not context.has_real_speech and "speech" in modalities and not context.rule_hints.explicit_speech_reference:
        modalities = [item for item in modalities if item != "speech"]
        if "visual" not in modalities:
            modalities.append("visual")
    temporal_order = analysis.temporal_relation in {"before_after", "sequence", "stage", "transition"}
    global_coverage = analysis.temporal_scope == "whole_video"
    structural_minimum = 2 if temporal_order or global_coverage else 1
    facts = [RequiredFact(fact=f"Establish from the video: {analysis.answer_target}", modality=item) for item in modalities]
    strict_scope = analysis.temporal_scope == "whole_video" and any(token in analysis.answer_target for token in ("完整", "全部", "所有", "一个都不要漏"))
    scope = ScopeRequirement(
        target="whole_video" if analysis.temporal_scope == "whole_video" else ("multi_event" if len(facts) > 1 else "local"),
        completeness="strict" if strict_scope else ("best_effort" if analysis.temporal_scope == "whole_video" else "none"),
        allow_partial_answer=not strict_scope,
    )
    return EvidencePlan(
        required_modalities=modalities,
        required_facts=facts,
        decision_facts=[analysis.answer_target] if analysis.answer_target else [],
        min_distinct_timestamps=structural_minimum,
        requires_temporal_order=temporal_order,
        requires_global_coverage=global_coverage,
        requires_visual_identity="visual" in modalities,
        human_readable_requirements=[fact.fact for fact in facts],
        reason="Conservative evidence plan derived from validated question analysis.",
        planning_method="fallback",
        claims=[ClaimRequirement(id=f"c{index}", description=fact.fact, preferred_modalities=[fact.modality], importance="core") for index, fact in enumerate(facts, start=1)],
        scope=scope,
    )


def _apply_constraints(plan: EvidencePlan, analysis: QuestionAnalysis, context: DeterministicQuestionContext) -> EvidencePlan:
    modalities = list(plan.required_modalities or analysis.content_modalities)
    if analysis.content_modalities:
        modalities = [item for item in modalities if item in analysis.content_modalities] or list(analysis.content_modalities)
    if not context.has_real_speech and not context.rule_hints.explicit_speech_reference:
        modalities = [item for item in modalities if item != "speech"]
        if not modalities:
            modalities = ["visual"]
    temporal_order = analysis.temporal_relation in {"before_after", "sequence", "stage", "transition"}
    global_coverage = plan.requires_global_coverage or analysis.temporal_scope == "whole_video"
    structural_minimum = 2 if temporal_order or global_coverage else 1
    facts = [fact for fact in plan.required_facts if fact.modality in modalities]
    if not facts:
        facts = [RequiredFact(fact=f"Establish from the video: {analysis.answer_target}", modality=item) for item in modalities]
    claims = plan.claims or [ClaimRequirement(id=f"c{index}", description=fact.fact, preferred_modalities=[fact.modality], importance="core") for index, fact in enumerate(facts, start=1)]
    scope = plan.scope
    if scope.target == "local" and global_coverage:
        scope = ScopeRequirement(target="whole_video", completeness="best_effort", allow_partial_answer=True, description="由全局范围兼容迁移生成。")
    return plan.model_copy(update={
        "required_modalities": modalities,
        "required_facts": facts,
        "min_distinct_timestamps": max(plan.min_distinct_timestamps, structural_minimum) if structural_minimum > 1 else 1,
        "requires_temporal_order": temporal_order,
        "requires_global_coverage": global_coverage,
        "claims": claims,
        "scope": scope,
    })
