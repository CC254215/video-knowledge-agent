# 已完成功能

## 输入与处理

- 支持通过公开视频 URL 处理视频。
- 支持上传或处理本地视频/音频文件。
- 支持读取视频元信息，包括标题、作者、时长、来源 URL。
- 支持优先使用字幕生成 transcript。
- 支持在无字幕时走本地 ASR 流程。
- 支持将处理结果保存到 `data/videos/{video_id}/`。
- 支持强制刷新旧缓存，重新处理视频证据。

## 多模态证据

- 支持使用 ffmpeg 从视频中抽取候选帧。
- 支持候选帧去重。
- 支持为每个多模态片段选择代表帧。
- 支持调用视觉模型生成 `frame_caption`。
- 支持在未配置视觉模型时标记 `missing_model`。
- 支持将 `speech`、`frame`、`frame_caption` 对齐到 `MultimodalSegment`。
- 支持生成并保存 `multimodal_segments.json`。
- 支持在 `multimodal_segments.json` 中记录：
  - 时间范围
  - speech 文本
  - frame ids
  - representative frame ids
  - frame captions
  - modality weight
  - evidence types
- 当前证据类型限定为：
  - `speech`
  - `frame`
  - `frame_caption`
- OCR 已按当前阶段要求禁用。

## 摘要与大纲

- 支持基于 `MultimodalSegment` 生成视频摘要。
- 支持生成 30 秒速览。
- 支持生成结构化大纲。
- 支持生成深度分析。
- 支持生成行动清单。
- 支持生成重要原文片段。
- 支持生成值得追问的问题。
- 支持记录 evidence coverage。
- 支持短视频单轮摘要。
- 支持长视频 map-reduce summary。
- 支持在摘要中记录：
  - `summary_strategy`
  - `num_chunks`
  - `chunk_summary_ids`
  - `generation_status`

## Query-Guided Storyline

- 支持生成 Query-Guided Storyline。
- 支持默认知识摄取 query。
- 支持按用户 query 生成 storyline。
- Storyline 节点支持：
  - `node_id`
  - `time_start`
  - `time_end`
  - `topic`
  - `claim`
  - `speech_evidence_ids`
  - `frame_evidence_ids`
  - `frame_caption_evidence`
  - `modality_support`
  - `uncertainty`
  - `status`
- 支持 Storyline 节点证据校验。
- 支持记录 Storyline validation warnings。
- 支持记录 Storyline evidence coverage。
- 支持记录 Storyline modality usage summary。

## 向量检索与证据选择

- 支持 ChromaDB 向量检索。
- 支持持久化 ChromaDB collection。
- 支持进程重启后继续检索已索引视频证据。
- 支持按 `video_id` 隔离检索结果。
- 支持索引 speech document。
- 支持索引 frame caption document。
- 支持不索引 OCR。
- 支持 embedding API。
- 支持 embedding 不可用时本地 hash embedding fallback。
- 支持 Query-DPP 证据选择。
- 支持 Evidence-DPP 证据选择。
- 支持计算 query/evidence agreement score。
- 支持根据 agreement score 输出置信度。

## 视频问答与对话

- 支持基于单个视频提问。
- 支持多轮视频对话。
- 支持在没有视频上下文时提示用户先处理视频。
- 支持问题类型识别。
- 支持根据问题判断是否需要视觉检查。
- 支持基于视频证据生成回答。
- 支持证据不足时明确拒绝编造。
- 支持回答中返回：
  - answer
  - evidence
  - timestamps
  - evidence_types
  - confidence
  - agreement_score
  - query_evidence_ids
  - answer_evidence_ids
  - dpp_used
  - needs_visual_check
  - reason
  - suggested_followup_questions
- 支持返回按时间排序的证据包。
- 证据包支持包含：
  - 时间段
  - speech 文本
  - frame caption
  - frame id
  - frame image path

## FrameEvidenceRefiner

- 支持在低置信度或视觉问题场景下触发补帧。
- 支持针对目标时间段补充抽帧。
- 支持对补充帧生成 frame caption。
- 支持补帧后更新 `multimodal_segments.json`。
- 支持补帧后更新 ChromaDB 索引。
- 支持补帧后重新回答。
- 支持每次问题最多自动补帧一次。
- 补帧流程不使用 OCR。

## Obsidian 导出

- 支持导出视频笔记到 Obsidian vault。
- 支持导出 Storyline 笔记到 Obsidian vault。
- 支持生成安全文件名。
- 支持避免覆盖已有文件。
- 支持在未配置 Obsidian vault 时跳过导出。
- 视频笔记支持包含：
  - 30 秒速览
  - 结构化大纲
  - 深度分析
  - 行动清单
  - Query-Guided Storyline
  - 关键证据
  - 多模态证据
  - 关键视觉帧
  - 用户对话记录
  - 原始证据索引

## 本地 Web UI

- 支持通过根目录 `cli.py` 启动本地 Web UI。
- 支持输入视频 URL。
- 支持上传本地视频/音频文件。
- 支持处理视频。
- 支持显示当前视频状态。
- 支持展示摘要。
- 支持展示 Storyline。
- 支持视频处理后在页面中提问。
- 支持展示最近一次回答的证据。
- 支持清空当前视频。
- 支持清空对话。

## CLI

- 支持 `process`。
- 支持 `ingest`。
- 支持 `summarize`。
- 支持 `ask`。
- 支持 `chat`。
- 支持 `export-obsidian`。
- 支持 `refine-frames`。
- 支持 `rebuild-index`。
- 支持 `audit`。
- 支持 `--no-ocr`。
- 支持 `--strict-llm`。
- 支持 `--force-refresh`。

## 配置与诊断

- 支持 `.env` 配置。
- 支持 OpenAI-compatible LLM API。
- 支持独立配置文本摘要模型。
- 支持独立配置视觉 caption 模型。
- 支持独立配置 embedding 模型。
- 支持启动时打印运行时模型配置。
- 支持自动移除 base URL 中误填的 `/chat/completions` 后缀。
- 支持检测并警告错误的 coding endpoint。
- 支持 strict LLM 模式。
- 支持 LLM smoke test。
- 支持 VLM smoke test。
- 支持 runtime config 检查脚本。

## 测试

- 已覆盖文本分段测试。
- 已覆盖多模态片段对齐测试。
- 已覆盖 frame caption pipeline 测试。
- 已覆盖 summary grounding 测试。
- 已覆盖 Storyline grounding 测试。
- 已覆盖 ChromaDB 检索测试。
- 已覆盖 ChromaDB 持久化测试。
- 已覆盖 DPP 测试。
- 已覆盖 FrameEvidenceRefiner 闭环测试。
- 已覆盖视频问答证据契约测试。
- 已覆盖 OCR 禁用策略测试。
- 已覆盖 Web UI 状态测试。
- 已覆盖上传文件校验测试。
- 已覆盖 runtime config 解析测试。
