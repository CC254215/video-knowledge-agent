"""Deterministic routing for cheap current-video factual questions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QAExecutionPlan:
    mode: Literal["fast", "full"]
    use_existing_speech: bool = True
    use_existing_visual: bool = True
    use_ocr: bool = True
    run_question_analysis: bool = True
    run_evidence_planner: bool = True
    run_memory_route: bool = True
    allow_fact_verification: bool = True
    allow_refinement: bool = True
    reason: str = ""


_CROSS_VIDEO = re.compile(r"上一个视频|之前那个视频|其他视频|跨视频|刚才另一个|以前问过|相比.*视频|another video|previous video", re.I)
_COMPLEX = re.compile(r"为什么|为何|怎么理解|原因|先.+再|先后|顺序|阶段|整个视频|全片|全程|总结|概括|核心|比较|对比|区别|前后|流程|过程|为什么|why|before|after|sequence|stage|whole video|compare", re.I)
_UI_DETAIL = re.compile(r"按钮|界面|点击|屏幕|画面|图中|截图|代码|公式|on[- ]screen|button|click", re.I)
_ENTITY_FACT = re.compile(r"什么软件|哪个软件|什么工具|哪个工具|什么模型|哪个模型|什么库|哪个库|什么产品|哪个产品|什么版本|哪个版本|使用了什么|用了什么|提到了什么", re.I)


def route_question(question: str) -> QAExecutionPlan:
    text = question.strip()
    if not text or _CROSS_VIDEO.search(text):
        return QAExecutionPlan("full", reason="empty_or_cross_video_scope")
    if _COMPLEX.search(text) or _UI_DETAIL.search(text):
        return QAExecutionPlan("full", reason="temporal_explanation_comparison_or_explicit_visual_detail")
    if _ENTITY_FACT.search(text):
        return QAExecutionPlan(
            "fast",
            run_question_analysis=False,
            run_evidence_planner=False,
            run_memory_route=False,
            reason="simple_current_video_entity_fact",
        )
    # Conservative default: preserve the complete reasoning chain.
    return QAExecutionPlan("full", reason="not_high_confidence_simple_fact")
