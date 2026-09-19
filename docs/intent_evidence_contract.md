# 问题理解与证据规划

当前问答入口使用以下链路：

```text
User Question
    ↓
Deterministic Parser
    ↓
Question Analyzer → QuestionAnalysis
    ↓
Evidence Planner → EvidencePlan
    ↓
Retrieval Planner → RetrievalPlan
    ↓
Retrieval / Evidence Contract
```

对应实现：

```text
app/reasoning/question_parser.py                Deterministic Parser
app/reasoning/question_analyzer.py              Question Analyzer
app/reasoning/evidence_requirement_planner.py   Evidence Planner
app/reasoning/retrieval_planner.py              Retrieval Planner
app/reasoning/intent_resolver.py                 协调与 legacy adapter
```

## 职责边界

### Deterministic Parser

只提取可确定的结构和环境事实：

- 明确时间戳及明确起止区间
- 整个视频等明确范围表达
- 选择题选项
- 是否存在真实转录以及静音占位文本
- 明确的画面、屏幕、说话或转录引用
- 低置信度的时间关系提示

普通冒号不会生成时间点。`process` 不会生成时间关系。时间关系提示不会覆盖模型分析。

### QuestionAnalysis

只描述用户在问什么：

```text
task_type
temporal_scope
temporal_relation
answer_target
content_modalities
choice_based
reason
analysis_method
```

`temporal_scope` 和 `temporal_relation` 是正交维度。顺序问题可以依赖语音、视觉或二者，不会自动触发视觉。

### EvidencePlan

描述可靠回答需要证明什么：

```text
required_modalities
required_facts
decision_facts
min_distinct_timestamps
requires_temporal_order
requires_global_coverage
requires_visual_identity
human_readable_requirements
reason
planning_method
```

控制流程依赖结构化字段；自然语言要求只用于 trace 和调试。当前项目没有 OCR 证据类型，因此本层使用 `speech` 和 `visual`，兼容层把 `visual` 映射为 `frame_caption`。

### RetrievalPlan

由 `required_facts`、`decision_facts` 和当前证据生成查询。Question Analyzer 不生成查询。后续补证轮次可以把新增证据重新传入 Retrieval Planner，从尚未覆盖的事实生成下一轮查询。

## 硬约束和软提示

- 明确时间戳、明确区间和媒体可用性是硬事实。
- before、after、phase 等词产生低置信度提示，LLM 可以纠正。
- 静音占位文本不进入转录预览，也不能满足 speech requirement。
- 无真实语音时，非语音问题会移除错误的 speech 模态；明确询问说话内容时保留 speech requirement，使下游能够判定不可回答。

## 兼容层

`ResolvedIntent` 暂时保留，供 `ConversationAgent`、Memory Gate 和回答 Prompt 使用。它由 `QuestionAnalysis + EvidencePlan` 单向生成：

- `question_type ← task_type`
- `required_evidence_types ← required_modalities`
- `needs_visual_check ← visual in required_modalities`
- `distinguishing_facts ← decision_facts`

兼容字段不会反向修改新模型。`retrieval_queries` 在兼容对象中为空，检索查询由独立的 Retrieval Planner 生成。

## Trace

每次持久化问答分别记录：

```text
deterministic_question_context
question_analysis
evidence_plan
retrieval_plan
evidence_contract_check
evidence_acquisition_plan
```

这允许区分语义理解、证据需求、检索计划、充分性检查和补证动作。

## 后续补证接口

`ConversationTurn` 已携带 `question_analysis` 和 `evidence_plan`。后续多轮闭环可以按以下顺序执行：

1. 对照 `required_facts` 和结构化约束检查缺口。
2. 把缺口与已有证据传给 Retrieval Planner。
3. 为尚未证明的事实生成新的检索或视觉采集动作。
4. 达到充分性、无新增证据或预算上限时停止。

当前提交没有修改最终 Answer Generator 的答案策略，也没有改造视频预处理和多轮补帧循环。
