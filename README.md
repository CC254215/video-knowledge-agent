# Video Knowledge Agent

**English:** A local-first pipeline that turns long-form video into structured, citeable knowledge: multimodal segments, summaries, query-guided storylines, hybrid retrieval, and grounded Q&A—without treating the project as a generic “video summarizer.”

**中文：** 本地优先的视频知识化 Agent：将视频处理为可追溯的多模态证据、摘要与叙事线，支持向量检索与有据可依的问答；当前阶段以 **`speech`（字幕/ASR）**、**`frame`（关键帧）**、**`frame_caption`（视觉描述）** 三类证据为主，**OCR 已按策略关闭**。

---

## Table of contents

- [Capabilities](#capabilities)
- [Architecture snapshot](#architecture-snapshot)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Typical workflows](#typical-workflows)
- [Optional integrations](#optional-integrations)
- [Artifacts & data layout](#artifacts--data-layout)
- [Throughput, retries & resume](#throughput-retries--resume)
- [Verification & diagnostics](#verification--diagnostics)
- [Development](#development)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)

---

## Capabilities

| Area | What you get |
|------|----------------|
| **Ingestion** | Public URLs (via yt-dlp), local video/audio files; metadata and transcript-first strategy with optional **local ASR** (`faster-whisper`, optional extra). |
| **Multimodal evidence** | Segment-aligned speech, representative frames, VLM captions; deduping and modality routing; artifacts persisted under `data/videos/{video_id}/`. |
| **Understanding** | LLM summaries (incl. map–reduce for long videos), 30s overview, structured outline, query-guided **Storyline** with grounded node checks. |
| **Retrieval** | ChromaDB persistent store (default under `data/chroma/`), hybrid / DPP-style evidence selection, agreement signals for confidence. |
| **Interfaces** | **Gradio** Web UI (`cli.py`), **Typer** CLI (`app.cli` / `src.cli` shim), **FastAPI** server (`api.py`). |

---

## Architecture snapshot

```text
URL / file → ingest (yt-dlp + ffmpeg) → transcript & segments
    → keyframes + VLM captions → multimodal_segments.json
    → embed & index (Chroma) → summary + storyline
    → QA / chat (evidence-bound answers)
```

Design principle: **answers must be supported by on-video evidence**; when evidence is insufficient, the system should state that explicitly rather than inventing facts.

---

## Requirements

- **Python** 3.11+（若系统默认 `python` 指向 3.10，请使用 `py -3.11`。）
- **ffmpeg** 在 `PATH` 中（真实 URL 下载与音视频处理依赖）。
- **yt-dlp**（已列入 `pyproject.toml` 依赖，随项目安装即可；命令行工具亦可单独维护）。
- **可选：** `faster-whisper`（无字幕时的本地 ASR）：`pip install -e ".[asr]"`。

项目若存在本地 ffmpeg  bundle，可将二进制置于例如：

```text
<project_root>/tools/ffmpeg/bin
```

并将该路径加入 `PATH`（与旧版说明一致，便于 Windows 下一键就绪）。

---

## Installation

```powershell
cd <project_root>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install -U pip
py -3.11 -m pip install -e .
# 可选：本地 ASR
py -3.11 -m pip install -e ".[asr]"
# 可选：跑测试
py -3.11 -m pip install -e ".[dev]"
```

安装后可使用控制台入口（见 `pyproject.toml`）：

- `vka` → Typer CLI（`app.cli`）
- `vka-api` → HTTP API（`api.main`）

---

## Quick start

### Web UI（Gradio）

```powershell
py -3.11 cli.py
```

常用参数：

```powershell
py -3.11 cli.py --port 7861
py -3.11 cli.py --share
py -3.11 cli.py --debug
```

默认地址：`http://127.0.0.1:7860`

### HTTP API（FastAPI / Uvicorn）

```powershell
py -3.11 api.py --host 127.0.0.1 --port 8000
```

或使用：`vka-api --port 8000`

### CLI（Typer）

`python -m app.cli` 与 `python -m src.cli` 等价（`src.cli` 为薄封装）。

示例：

```powershell
py -3.11 -m app.cli process --url "<video_url>" --no-ocr --strict-llm
py -3.11 -m app.cli ask --video-id "<video_id>" --question "视频的核心论点是什么？"
py -3.11 -m app.cli rebuild-index --video-id "<video_id>"
py -3.11 -m app.cli refine-frames --video-id "<video_id>" --start 600 --end 720
py -3.11 -m app.cli show-config
```

`process` 支持本地文件：`--file path\to\video.mp4`；更多标志见 `py -3.11 -m app.cli process --help`。

---

## Configuration

1. 复制环境变量模板：

   ```powershell
   copy .env.example .env
   ```

2. 在 `.env` 中填写 **LLM / VLM / Embedding** 的供应商 URL、模型名与密钥；**切勿**将 `.env` 提交到版本库。

3. 完整键名与默认值以 **`.env.example`** 为准（项目迭代时以仓库内文件为单一事实来源）。

**推荐起步配置示例（智谱 BigModel，可按需替换）：**

```env
LLM_PROVIDER=zhipu
LLM_API_KEY=${ZHIPU_API_KEY}
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_SUMMARY_MODEL=glm-4.6

VLM_PROVIDER=zhipu
VLM_API_KEY=${ZHIPU_API_KEY}
VLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
VLM_CAPTION_MODEL=GLM-4.6V-FlashX

EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=Embedding-3

DATA_DIR=./data
ASR_MODEL=small

ENABLE_OCR=false
EVIDENCE_TYPES=speech,frame,frame_caption
ALLOW_CODING_ENDPOINT_FOR_RUNTIME=false
REQUEST_TIMEOUT_SECONDS=600
```

**Runtime / Storyline 摘要相关（可按需调优）：**

```env
STRICT_LLM=false
SUMMARY_MAX_SEGMENTS=24
SUMMARY_SEGMENT_CHARS=500
SUMMARY_CAPTION_CHARS=300
STORYLINE_TOP_K=16
```

### LLM endpoint 安全提示

- **不要**将面向「代码 / Coding」场景的 OpenAPI 端点用于本项目的运行时推理（例如包含 `coding` 路径的 URL）。
- 若检测到 runtime LLM 指向 coding 端点，运行时会打印警告；设置 **`STRICT_LLM=true`**（或在 CLI 使用 `--strict-llm`）时可能直接失败退出，以避免误用。

---

## Typical workflows

### 处理公开 URL

```powershell
py -3.11 -m app.cli process --url "<公开视频_URL>" --no-ocr --strict-llm
```

成功时 CLI 会汇总管线状态、摘要生成状态、Chroma 索引统计、阶段耗时以及摘要 / Storyline 片段预览。

### 单视频问答

进程重启后如需仅重建向量索引：

```powershell
py -3.11 -m app.cli rebuild-index --video-id "<video_id>"
py -3.11 -m app.cli ask --video-id "<video_id>" --question "……"
```

回答结构通常包含：`answer`、`evidence`、`timestamps`、`evidence_types`、`confidence`、`agreement_score`、`needs_visual_check`、`reason` 等（以当前模型与管线版本为准）。

---

## Optional integrations

### Obsidian

配置 **`OBSIDIAN_VAULT_PATH`** 后，处理完成可尝试导出（具体路径规则见实现与 `COMPLETED_FEATURES.md`），例如：

- `10_Sources/Videos/{date} - {safe_title}.md`
- `30_Storylines/{date} - {safe_title} - storyline.md`

未配置 vault 时跳过导出，不阻塞主流程。

### MemPalace（MCP stdio）

默认关闭。启用后与 MemPalace MCP server 通信，用于**长期记忆补充**，不替代视频证据。

```env
MEMPALACE_PROVIDER=mcp_stdio
MEMPALACE_COMMAND=python
MEMPALACE_ARGS=-m mempalace.mcp_server
MEMPALACE_PALACE_PATH=
MEMPALACE_TIMEOUT_SECONDS=30
MEMPALACE_AUTO_STATUS=true
```

调试示例：

```powershell
py -3.11 scripts\smoke_test_mempalace_mcp.py
py -3.11 -m app.cli mempalace-status
py -3.11 -m app.cli mempalace-search "agent memory"
```

边界：**视频事实仍须由 `speech` / `frame` / `frame_caption` 支持**；MemPalace 结果以结构化 **`long_term_memory`** 等形式注入上下文，不会冒充视频内证据。

---

## Artifacts & data layout

处理产物默认位于：

```text
data/videos/{video_id}/
```

常见核心文件包括：

- `metadata.json`
- `transcript.json`
- `multimodal_segments.json`
- `summary.json`
- `storyline.json`
- `llm_logs/`

向量库默认持久化目录可参考 `.env` 中的 **`CHROMA_PATH`**（示例中为 `./data/chroma`）。

---

## Throughput, retries & resume

对外部依赖的请求（LLM、VLM、Embedding、yt-dlp、云 ASR 等）由调度参数限制并发与最小间隔，可在 `.env` 中调节，例如：

```env
LLM_MAX_CONCURRENCY=2
LLM_MIN_INTERVAL_SECONDS=1.5
VLM_CAPTION_CONCURRENCY=3
VLM_MIN_INTERVAL_SECONDS=2.0
EMBEDDING_MAX_CONCURRENCY=1
EMBEDDING_MIN_INTERVAL_SECONDS=1.0
YTDLP_MAX_CONCURRENCY=1
YTDLP_MIN_INTERVAL_SECONDS=6.0
ASR_MAX_CONCURRENCY=1
ASR_MIN_INTERVAL_SECONDS=2.0
API_MAX_RETRIES=3
API_RETRY_BASE_DELAY_SECONDS=2.0
PIPELINE_RESUME=true
PIPELINE_FORCE_STAGE=
```

每个视频目录会维护 **`processing_state.json`**。再次运行 `process` 时，默认跳过已成功且产物存在的阶段；需要全量重跑可使用 `--force-refresh`，或：

```powershell
py -3.11 -m app.cli process --url "<video_url>" --force-refresh
py -3.11 -m app.cli process --url "<video_url>" --force-stage summary
py -3.11 -m app.cli process --url "<video_url>" --no-resume
```

---

## Verification & diagnostics

| 目的 | 命令 |
|------|------|
| 检查当前解析到的运行时配置 | `py -3.11 scripts\check_runtime_config.py` |
| 文本摘要模型冒烟 | `py -3.11 scripts\smoke_test_llm.py` |
| 视觉 caption 模型冒烟 | `py -3.11 scripts\smoke_test_vlm.py` |

上述脚本会输出 provider、base URL、model、HTTP 状态与响应正文片段，**不会打印 API key**。

---

## Development

```powershell
py -3.11 -m pytest
```

测试配置见 `pyproject.toml` 中 `[tool.pytest.ini_options]`；功能清单见 **`COMPLETED_FEATURES.md`**。

---

## Limitations

- **平台与站点策略**：下载成功率依赖 yt-dlp 与各站点策略变更。
- **ffmpeg**：真实 URL 与转码路径依赖本机 ffmpeg 可用。
- **视觉理解**：当前阶段以关键帧 + caption 为主，非完整像素级视频理解。
- **OCR**：按策略关闭；启用需依赖可选组件并自行评估成本与效果。
- **ASR 质量**：本地模型与音源质量直接影响转写效果。
- **向量库**：持久化路径需自行备份；重建索引仍可从 JSON 等产物恢复索引内容（参见 `rebuild-index`）。

---

## Repository layout

| Path | Role |
|------|------|
| `app/` | 主包：配置、管线、检索、推理、UI、API、vision、storage 等 |
| `src/` | `python -m src.cli` 入口封装，转发至 `app.cli` |
| `cli.py` | 启动 Gradio Web UI |
| `api.py` | 启动 FastAPI 应用 |
| `scripts/` | 配置检查与冒烟脚本 |
| `tests/` | Pytest 用例 |

---

## Version

当前包版本见 `pyproject.toml` 中 `version`（例如 `0.1.0`）。
