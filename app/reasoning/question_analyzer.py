from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import DeterministicQuestionContext, QuestionAnalysis
from app.reasoning.llm_client import LLMClient


def analyze_question(
    question: str,
    context: DeterministicQuestionContext,
    settings: Settings | None = None,
    log_dir: Path | None = None,
    use_llm: bool = True,
) -> QuestionAnalysis:
    fallback = _fallback_question_analysis(question, context)
    if not use_llm or not settings or not settings.has_llm_config:
        return fallback
    try:
        payload = LLMClient(settings, log_dir).generate_json(
            build_question_analyzer_prompt(question, context), schema_hint=question_analysis_schema()
        )
        analysis = QuestionAnalysis.model_validate({**payload, "analysis_method": "llm"})
    except Exception:
        return fallback
    return _ensure_output_language(_apply_hard_constraints(analysis, context), question)


def _ensure_output_language(analysis: QuestionAnalysis, question: str) -> QuestionAnalysis:
    """Keep Chinese user questions in a Chinese intent representation."""
    if not any("\u4e00" <= char <= "\u9fff" for char in question):
        return analysis
    has_cjk = lambda value: any("\u4e00" <= char <= "\u9fff" for char in value)
    updates: dict[str, Any] = {}
    if not has_cjk(analysis.answer_target):
        updates["answer_target"] = question.strip()[:500]
    if not has_cjk(analysis.reason):
        updates["reason"] = "已按中文问题识别其任务类型、时间范围和所需证据模态。"
    return analysis.model_copy(update=updates) if updates else analysis


def build_question_analyzer_prompt(question: str, context: DeterministicQuestionContext) -> str:
    environment = {
        "video_title": context.video_title,
        "video_duration": context.video_duration,
        "has_real_speech": context.has_real_speech,
        "has_audio": context.has_audio,
        "explicit_timestamps": [item.model_dump() for item in context.explicit_timestamps],
        "explicit_intervals": [item.model_dump() for item in context.explicit_intervals],
        "explicit_whole_video": context.explicit_whole_video,
        "choice_based": context.choice_based,
        "choices": context.choices,
        "transcript_preview": context.transcript_preview,
        "rule_hints": context.rule_hints.model_dump(),
    }
    return f"""
你负责分析视频问答中的用户问题。

你的任务只是理解“用户在问什么”，不要回答问题，不要搜索证据，也不要决定是否需要补帧。

请判断：
1. 用户希望得到什么类型的答案。
2. 问题涉及的视频时间范围。
3. 是否要求判断事件之间的时间关系。
4. 问题主要询问说话内容、视觉内容、二者结合，还是目前无法预先确定。
5. 问题是否是多选题。

时间范围与时间关系必须分开。不要因为 process、after、phase 等单个词机械分类，要根据完整语义判断。
whole_video 只用于问题明确要求全片、总体流程或必须纵览视频才能回答的情况。概念本身具有步骤不代表视频时间范围是 whole_video；没有视频范围要求时使用 none。
content_modalities 只能使用 speech、visual；如果无法判断，可以返回空数组。
不要生成 retrieval query。不要判断当前证据是否充分。不要决定是否补帧。不要回答用户问题。
rule_hints 是低置信度提示，可以纠正；explicit_timestamps、explicit_intervals 和 explicit_whole_video 是硬事实。

输出语言要求：answer_target 和 reason 必须使用简体中文；不要把用户的中文问题翻译成英文，也不要输出英文检索词。

确定性环境信息：
{json.dumps(environment, ensure_ascii=False, indent=2)}

用户问题：
{question}
"""


def question_analysis_schema() -> str:
    return """
{
  "task_type": "factual|summary|explanation|comparison|evidence_check|critique|action_items|procedure",
  "temporal_scope": "none|point|interval|event_local|whole_video",
  "temporal_relation": "none|before_after|sequence|stage|transition",
  "answer_target": "what the user wants established",
  "content_modalities": ["speech", "visual"],
  "choice_based": false,
  "reason": "short semantic reason"
}
"""


def _fallback_question_analysis(question: str, context: DeterministicQuestionContext) -> QuestionAnalysis:
    scope = "interval" if context.explicit_intervals else "point" if context.explicit_timestamps else "whole_video" if context.explicit_whole_video else "none"
    relation = context.rule_hints.possible_temporal_relation
    if scope == "none" and relation == "before_after":
        scope = "event_local"
    modalities: list[str] = []
    if context.rule_hints.explicit_speech_reference:
        modalities.append("speech")
    if context.rule_hints.explicit_visual_reference:
        modalities.append("visual")
    if not modalities:
        modalities = ["speech"] if context.has_real_speech else ["visual"]
    return QuestionAnalysis(
        task_type=_fallback_task_type(question),  # type: ignore[arg-type]
        temporal_scope=scope,  # type: ignore[arg-type]
        temporal_relation=relation,
        answer_target=_clean_answer_target(question),
        content_modalities=modalities,  # type: ignore[arg-type]
        choice_based=context.choice_based,
        reason="LLM unavailable; conservative fallback from deterministic facts and explicit references.",
        analysis_method="fallback",
    )


def _apply_hard_constraints(analysis: QuestionAnalysis, context: DeterministicQuestionContext) -> QuestionAnalysis:
    updates: dict[str, Any] = {"choice_based": context.choice_based or analysis.choice_based}
    if context.explicit_intervals:
        updates["temporal_scope"] = "interval"
    elif context.explicit_timestamps:
        updates["temporal_scope"] = "point"
    elif context.explicit_whole_video:
        updates["temporal_scope"] = "whole_video"
    modalities = list(analysis.content_modalities)
    if context.rule_hints.explicit_speech_reference and "speech" not in modalities:
        modalities.insert(0, "speech")
    if not context.has_real_speech and not context.rule_hints.explicit_speech_reference:
        modalities = [item for item in modalities if item != "speech"]
        if not modalities:
            modalities = ["visual"]
    updates["content_modalities"] = modalities
    return analysis.model_copy(update=updates)


def _fallback_task_type(question: str) -> str:
    patterns = (
        ("summary", r"总结|概括|主要内容|\bsummar(?:y|ize)\b|\boverview\b"),
        ("evidence_check", r"证据|依据|是否支持|证明|\bevidence\b|\bsupport\b"),
        ("comparison", r"比较|对比|区别|\bcompar(?:e|ison)\b|\bdifference\b"),
        ("critique", r"批判|局限|漏洞|反驳|\bcritique\b|\blimitation\b"),
        ("action_items", r"行动建议|操作建议|风控建议|清单|\baction items?\b|\brecommendations?\b"),
        ("procedure", r"步骤|流程|阶段|如何制作|\bprocedure\b|\bsteps?\b"),
        ("explanation", r"为什么|解释|怎么理解|\bwhy\b|\bexplain\b"),
    )
    return next((name for name, pattern in patterns if re.search(pattern, question, re.I)), "factual")


def _clean_answer_target(question: str) -> str:
    return " ".join(re.split(r"(?im)^\s*choices\s*[:：]\s*$", question, maxsplit=1)[0].split())[:500]
