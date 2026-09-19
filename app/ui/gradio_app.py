from __future__ import annotations

import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from html import escape
from typing import Any

from app.services.aggregation_service import AggregationService
from app.services.chat_service import ChatAnswerResult, ChatService
from app.services.streaming_video_service import StreamingVideoService, parse_url_lines
from app.services.video_service import VideoService
from app.ui.formatters import (
    format_chat_answer_for_ui,
    format_evidence_for_ui,
    format_evidence_gallery_items,
    format_status_for_ui,
    format_storyline_for_ui,
    format_suggested_questions_for_ui,
    format_summary_for_ui,
)
from app.ui.state import AppState

STAGE_PROGRESS = {
    "解析": 10,
    "URL": 10,
    "复制": 12,
    "字幕": 25,
    "ASR": 30,
    "语音识别": 38,
    "下载视频": 10,
    "下载音频": 30,
    "读取视频信息": 6,
    "加载元数据": 55,
    "关键帧": 58,
    "检索索引": 70,
    "生成摘要": 78,
    "生成 storyline": 88,
    "保存处理结果": 95,
    "完成处理": 98,
    "抽帧": 45,
    "多模态": 60,
    "摘要": 72,
    "storyline": 84,
    "检索": 92,
    "Obsidian": 97,
    "完成": 100,
}

THEME_CSS = """
:root {
  --vka-orange: #ff7a00;
  --vka-orange-soft: #fff4ea;
  --vka-border: #e7ebf2;
  --vka-text: #1f2937;
  --vka-muted: #667085;
  --vka-bg: #f7f9fc;
}
.gradio-container { background: radial-gradient(circle at top left, #fff7ed 0, #f8fafc 280px, #f7f9fc 100%); color: var(--vka-text); }
#vka-root { max-width: 1680px; margin: 0 auto; }
.vka-header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin: 8px 0 18px; }
.brand { display:flex; align-items:center; gap:12px; font-size:28px; font-weight:800; }
.brand-icon { width:42px; height:42px; border-radius:50%; background:linear-gradient(135deg,#ff9d2e,#ff6b00); color:white; display:grid; place-items:center; box-shadow:0 8px 20px #ff7a0033; }
.brand-subtitle { color:var(--vka-muted); font-size:14px; font-weight:500; }
.panel { border:1px solid var(--vka-border); border-radius:14px; background:rgba(255,255,255,.9); box-shadow:0 8px 24px rgba(15,23,42,.04); padding:14px; }
.soft-panel { background: var(--vka-orange-soft); border:1px solid #ffe2c4; border-radius:12px; padding:12px; }
.section-title { font-weight:750; margin:12px 0 8px; font-size:15px; }
.empty-card { min-height:110px; display:grid; place-items:center; color:var(--vka-muted); border:1px dashed var(--vka-border); border-radius:12px; background:#fbfcff; padding:18px; text-align:center; }
.status-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; }
.status-grid div { border:1px solid var(--vka-border); border-radius:10px; padding:10px; background:#fff; }
.status-grid span { display:block; color:var(--vka-muted); font-size:12px; margin-bottom:4px; }
.status-grid b { font-size:13px; word-break:break-word; }
.error-card { background:#fff1f2; border:1px solid #fecdd3; border-radius:12px; padding:12px; color:#be123c; }
.progress-shell { border:1px solid var(--vka-border); background:#fff; border-radius:8px; padding:12px; }
.progress-head { display:flex; justify-content:space-between; gap:12px; margin-bottom:8px; font-size:13px; }
.progress-track { height:10px; overflow:hidden; border-radius:5px; background:#e9edf3; }
.progress-fill { height:100%; background:#ff7a00; transition:width .25s ease; }
.timing-panel { margin-top:12px; border-top:1px solid var(--vka-border); padding-top:12px; }
.timing-title { display:flex; justify-content:space-between; gap:12px; margin-bottom:10px; font-size:13px; }
.timing-title span { color:var(--vka-muted); }
.timing-row { display:grid; grid-template-columns:minmax(130px,1.5fr) minmax(100px,3fr) 82px; align-items:center; gap:10px; margin:7px 0; font-size:12px; }
.timing-label { overflow-wrap:anywhere; }
.timing-track { height:7px; overflow:hidden; border-radius:4px; background:#e9edf3; }
.timing-fill { display:block; height:100%; background:#ff7a00; }
.timing-fill.failed_retryable,.timing-fill.failed_terminal { background:#dc2626; }
.timeline { display:flex; flex-direction:column; gap:10px; }
.timeline-node { display:flex; gap:10px; border:1px solid var(--vka-border); border-radius:12px; padding:10px; background:#fff; }
.timeline-dot { width:10px; height:10px; min-width:10px; border-radius:50%; background:var(--vka-orange); margin-top:6px; }
.time-chip,.type-chip,.score-chip,.status-chip { display:inline-block; border-radius:999px; padding:2px 8px; font-size:12px; margin-right:6px; }
.time-chip { background:#f2f4f7; color:#475467; }
.type-chip { background:#eef4ff; color:#1d4ed8; }
.score-chip { background:#ecfdf3; color:#087443; }
.status-chip { margin-top:6px; background:#fff7ed; color:#c2410c; }
.muted { color:var(--vka-muted); font-size:12px; }
.small { font-size:11px; margin-top:10px; }
.bullet,.sub-bullet { margin:6px 0; line-height:1.55; }
.sub-bullet { color:#475467; margin-left:12px; }
.outline-item { margin-top:10px; }
.question-pill { width:100%; text-align:left; border:1px solid var(--vka-border); background:#fff; border-radius:10px; padding:9px 10px; margin:4px 0; color:#344054; }
.evidence-list { display:flex; flex-direction:column; gap:10px; max-height:520px; overflow:auto; padding-right:4px; }
.evidence-card { border:1px solid var(--vka-border); border-radius:12px; padding:10px; background:#fff; }
.evidence-head { display:flex; align-items:center; gap:4px; margin-bottom:6px; }
.evidence-text { margin-top:8px; line-height:1.55; font-size:13px; color:#344054; }
#send-btn button, #process-btn button { background:linear-gradient(135deg,#ff8a00,#ff6500)!important; border:none!important; color:white!important; }
#right-rail { position: sticky; top: 12px; align-self:flex-start; }
"""


def build_app(debug: bool = False):
    import gradio as gr

    video_service = VideoService()
    chat_service = ChatService()
    aggregation_service = AggregationService()
    streaming_service = StreamingVideoService()
    initial_result = video_service.load_latest_completed()
    initial_state = AppState()
    if initial_result:
        initial_state.load_video(
            initial_result.video_id or "",
            metadata=initial_result.metadata,
            summary=initial_result.summary,
            storyline=initial_result.storyline,
            modality_profile=initial_result.modality_profile,
        )

    with gr.Blocks(title="Video Knowledge Agent") as demo:
        state = gr.State(initial_state)
        with gr.Column(elem_id="vka-root"):
            gr.HTML(
                """
                <div class="vka-header">
                  <div>
                    <div class="brand"><span class="brand-icon">▶</span> Video Knowledge Agent</div>
                    <div class="brand-subtitle">Multimodal video understanding, evidence tracing, and interactive QA</div>
                  </div>
                  <div class="brand-subtitle">本地优先 · 证据可追溯 · OCR 当前禁用</div>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            video_url = gr.Textbox(label="Video URL", placeholder="https://...", elem_id="video-url")
                            gr.HTML("<div class='soft-panel'><b>当前视频状态</b><br><span class='muted'>输入 URL 或上传文件后点击处理。</span></div>")
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            video_file = gr.File(label="Upload video/audio", type="filepath")
                            gr.HTML("<div class='empty-card'>支持 MP4 / MOV / MKV / WEBM / MP3 / WAV / M4A</div>")

                    with gr.Row():
                        process_btn = gr.Button("处理视频", variant="primary", elem_id="process-btn")
                        aggregate_btn = gr.Button("聚合当前视频")
                        clear_video_btn = gr.Button("清空当前视频")

                    with gr.Column(elem_classes=["panel"]):
                        gr.Markdown("### 流式输入")
                        stream_urls = gr.Textbox(
                            label="Batch Video URLs",
                            placeholder="每行一个 URL。阶段一完成后会立即进入下一个 URL，后续分析在后台继续。",
                            lines=4,
                        )
                        stream_btn = gr.Button("流式处理队列")
                        stream_status_box = gr.JSON(label="Streaming Queue")
                        aggregate_status_box = gr.JSON(label="Cross-video Aggregation")

                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            progress_bar = gr.HTML(_progress_html(100, "已恢复最近完成的视频") if initial_result else _progress_html(0, "等待输入"))
                            status_box = gr.HTML(format_status_for_ui(initial_result))
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            summary_box = gr.HTML(format_summary_for_ui(initial_result.summary if initial_result else None), label="摘要")

                    with gr.Row():
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            suggested_questions_box = gr.HTML(
                                format_suggested_questions_for_ui(initial_result.summary if initial_result else None),
                                label="建议追问",
                            )
                        with gr.Column(scale=1, elem_classes=["panel"]):
                            storyline_box = gr.HTML(format_storyline_for_ui(initial_result.storyline if initial_result else None), label="Storyline")

                    with gr.Column(elem_classes=["panel"]):
                        gr.Markdown("### Video Chat")
                        chatbot = gr.Chatbot(label="Video Chat", height=420)
                        with gr.Row():
                            user_question = gr.Textbox(label="Ask about this video", placeholder="围绕当前视频提问", scale=5)
                            send_btn = gr.Button("发送", variant="primary", scale=1, elem_id="send-btn")
                            clear_chat_btn = gr.Button("清空对话", scale=1)

                with gr.Column(scale=5, elem_id="right-rail"):
                    with gr.Column(elem_classes=["panel"]):
                        gr.Markdown("### Answer Evidence")
                        evidence_box = gr.HTML("<div class='empty-card'>回答后将在这里显示答案引用的文本片段、时间戳和帧证据。</div>")
                    with gr.Column(elem_classes=["panel"]):
                        gr.Markdown("### Evidence Frames")
                        evidence_gallery = gr.Gallery(
                            value=_gallery_from_result(initial_result) if initial_result else [],
                            label=None,
                            columns=2,
                            height=340,
                            object_fit="contain",
                        )
                    with gr.Column(elem_classes=["panel"]):
                        gr.Markdown("### Diagnostics")
                        diagnostics_box = gr.JSON(value=_diagnostics(initial_result) if initial_result else {}, label=None)
                        debug_box = gr.JSON(label="Debug Raw", visible=debug)

        process_btn.click(
            fn=partial(_process_video, video_service),
            inputs=[video_url, video_file, state],
            outputs=[
                state,
                progress_bar,
                status_box,
                summary_box,
                storyline_box,
                suggested_questions_box,
                chatbot,
                evidence_box,
                evidence_gallery,
                diagnostics_box,
                debug_box,
            ],
        )
        aggregate_btn.click(
            fn=lambda app_state: _aggregate_current_video(aggregation_service, app_state),
            inputs=[state],
            outputs=[aggregate_status_box, diagnostics_box, debug_box],
        )
        stream_btn.click(
            fn=partial(_stream_process_urls, streaming_service),
            inputs=[stream_urls, state],
            outputs=[
                state,
                progress_bar,
                status_box,
                summary_box,
                storyline_box,
                suggested_questions_box,
                chatbot,
                evidence_box,
                evidence_gallery,
                diagnostics_box,
                debug_box,
                stream_status_box,
            ],
        )
        send_btn.click(
            fn=lambda question, app_state: _send_message(chat_service, question, app_state),
            inputs=[user_question, state],
            outputs=[state, chatbot, evidence_box, evidence_gallery, suggested_questions_box, user_question, diagnostics_box, debug_box],
        )
        user_question.submit(
            fn=lambda question, app_state: _send_message(chat_service, question, app_state),
            inputs=[user_question, state],
            outputs=[state, chatbot, evidence_box, evidence_gallery, suggested_questions_box, user_question, diagnostics_box, debug_box],
        )
        clear_video_btn.click(
            fn=_clear_video,
            inputs=[state],
            outputs=[
                state,
                progress_bar,
                status_box,
                summary_box,
                storyline_box,
                suggested_questions_box,
                chatbot,
                evidence_box,
                evidence_gallery,
                diagnostics_box,
                debug_box,
            ],
        )
        clear_chat_btn.click(
            fn=_clear_chat,
            inputs=[state],
            outputs=[state, chatbot, evidence_box, evidence_gallery, suggested_questions_box, diagnostics_box, debug_box],
        )

    return demo


def _aggregate_current_video(service: AggregationService, app_state: AppState):
    result = service.aggregate_current_video(app_state.current_video_id)
    payload = _debug(result)
    diagnostics = {
        "aggregation_success": result.success,
        "current_video_included": result.current_video_included,
        "topic_count": result.topic_count,
        "record_count": result.record_count,
        "last_error": result.error,
    }
    return payload, diagnostics, payload


def _stream_process_urls(service: StreamingVideoService, urls_text: str | None, app_state: AppState):
    urls = parse_url_lines(urls_text)
    if not urls:
        yield (
            app_state,
            0,
            "<div class='error-card'>请至少输入一个 URL。</div>",
            format_summary_for_ui(app_state.current_summary),
            format_storyline_for_ui(app_state.current_storyline),
            format_suggested_questions_for_ui(app_state.current_summary),
            app_state.chat_history,
            "<div class='empty-card'>暂无新证据。</div>",
            [],
            {"last_error": "empty_stream_urls"},
            {},
            {"events": [], "results": []},
        )
        return

    last_success = None
    for batch_state in service.process_urls(urls):
        for result in batch_state.results:
            if result.status == "succeeded":
                last_success = result
        if last_success:
            app_state.load_video(
                last_success.video_id or "",
                metadata=last_success.metadata,
                summary=last_success.summary,
                storyline=last_success.storyline,
            )
        percent = int(((batch_state.completed + batch_state.failed) / max(1, batch_state.total)) * 100)
        message = f"流式队列：完成 {batch_state.completed}，失败 {batch_state.failed}，总数 {batch_state.total}"
        yield (
            app_state,
            _progress_html(percent, message),
            f"<div class='status-card'><b>{message}</b><br><span class='muted'>阶段一与后续分析分离执行。</span></div>",
            format_summary_for_ui(app_state.current_summary),
            format_storyline_for_ui(app_state.current_storyline),
            format_suggested_questions_for_ui(app_state.current_summary),
            app_state.chat_history,
            "<div class='empty-card'>流式队列完成单个视频后，可继续对当前成功视频提问。</div>",
            [],
            {
                "stream_total": batch_state.total,
                "stream_completed": batch_state.completed,
                "stream_failed": batch_state.failed,
                "last_error": next((item.error for item in reversed(batch_state.results) if item.error), None),
            },
            _stream_debug(batch_state),
            _stream_debug(batch_state),
        )


def _process_video(service: VideoService, url: str | None, file_path: str | None, app_state: AppState):
    progress_events: queue.Queue[tuple[int, str]] = queue.Queue()

    def progress_callback(message: str) -> None:
        progress_events.put((_progress_value(message), message))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.process_video, url, file_path, None, progress_callback)
        last_percent = 1
        started_at = time.monotonic()
        last_heartbeat = started_at
        yield _process_outputs(app_state, last_percent, "正在启动处理任务")
        while not future.done():
            try:
                percent, message = progress_events.get(timeout=0.25)
            except queue.Empty:
                now = time.monotonic()
                if now - last_heartbeat >= 1.0:
                    last_heartbeat = now
                    elapsed = int(now - started_at)
                    yield _process_outputs(app_state, last_percent, f"任务仍在运行，已用时 {elapsed}s；当前阶段可能正在等待模型或外部服务返回")
                continue
            last_percent = max(last_percent, percent)
            yield _process_outputs(app_state, last_percent, message)
        result = future.result()

    if not result.success:
        app_state.fail(result.error or "处理失败")
        yield (
            app_state,
            _progress_html(100, "处理失败"),
            format_status_for_ui(result),
            format_summary_for_ui(None),
            format_storyline_for_ui(None),
            "<div class='empty-card'>处理失败后暂无建议追问。</div>",
            [],
            "<div class='empty-card'>暂无证据。</div>",
            [],
            _diagnostics(result),
            _debug(result),
        )
        return

    app_state.load_video(
        result.video_id or "",
        metadata=result.metadata,
        summary=result.summary,
        storyline=result.storyline,
        modality_profile=result.modality_profile,
    )
    yield (
        app_state,
        _progress_html(100, "处理完成"),
        format_status_for_ui(result),
        format_summary_for_ui(result.summary),
        format_storyline_for_ui(result.storyline),
        format_suggested_questions_for_ui(result.summary),
        [],
        "<div class='empty-card'>视频处理完成。提问后这里会显示本次回答引用的证据。</div>",
        _gallery_from_result(result),
        _diagnostics(result),
        _debug(result),
    )


def _process_outputs(app_state: AppState, percent: int, message: str):
    return (
        app_state,
        _progress_html(percent, message),
        f"<div class='status-card'><b>{message}</b><br><span class='muted'>处理中，请保持页面打开。</span></div>",
        format_summary_for_ui(None),
        format_storyline_for_ui(None),
        "<div class='empty-card'>处理中...</div>",
        [],
        "<div class='empty-card'>暂无证据。</div>",
        [],
        {"events": [{"stage": message, "progress": percent}], "coverage": min(percent / 100, 0.99)},
        {},
    )


def _send_message(service: ChatService, question: str, app_state: AppState):
    question = (question or "").strip()
    if not question:
        return app_state, app_state.chat_history, "<div class='empty-card'>请输入问题。</div>", [], format_suggested_questions_for_ui(app_state.current_summary), "", {}, {}
    result = service.ask(app_state.current_video_id, question, app_state.current_conversation_id)
    answer_text = format_chat_answer_for_ui(result.raw) if result.raw else _format_service_error(result)
    app_state.chat_history.append({"role": "user", "content": question})
    app_state.chat_history.append({"role": "assistant", "content": answer_text})
    return (
        app_state,
        app_state.chat_history,
        format_evidence_for_ui(result.raw) if result.raw else "<div class='empty-card'>暂无视频证据。</div>",
        _gallery_from_evidence(result.evidence),
        format_suggested_questions_for_ui(app_state.current_summary, result.raw),
        "",
        _diagnostics(result),
        _debug(result),
    )


def _clear_video(app_state: AppState):
    app_state.clear_video()
    return (
        app_state,
        _progress_html(0, "等待输入"),
        format_status_for_ui(None),
        format_summary_for_ui(None),
        format_storyline_for_ui(None),
        format_suggested_questions_for_ui(None),
        [],
        "<div class='empty-card'>暂无证据。</div>",
        [],
        {},
        {},
    )


def _clear_chat(app_state: AppState):
    app_state.clear_chat()
    return app_state, [], "<div class='empty-card'>暂无证据。</div>", [], format_suggested_questions_for_ui(app_state.current_summary), {}, {}


def _format_service_error(result: ChatAnswerResult) -> str:
    return f"{result.answer}\n\n**置信度：** `{result.confidence}`\n\n**原因：** {result.reason}"


def _progress_value(message: str) -> int:
    match = re.search(r"(下载视频|下载音频|语音识别)\s+(\d+(?:\.\d+)?)%", message)
    if match:
        operation, raw_percent = match.groups()
        value = max(0.0, min(100.0, float(raw_percent)))
        ranges = {"下载视频": (10, 28), "下载音频": (30, 38), "语音识别": (38, 55)}
        start, end = ranges[operation]
        return round(start + (end - start) * value / 100)
    for token, value in STAGE_PROGRESS.items():
        if token.lower() in message.lower():
            return value
    return 50


def _progress_html(percent: int, message: str) -> str:
    value = max(0, min(100, int(percent)))
    safe_message = escape(message)
    return (
        "<div class='progress-shell'>"
        f"<div class='progress-head'><b>{value}%</b><span>{safe_message}</span></div>"
        f"<div class='progress-track' role='progressbar' aria-valuemin='0' aria-valuemax='100' "
        f"aria-valuenow='{value}'><div class='progress-fill' style='width:{value}%'></div></div>"
        "</div>"
    )


def _gallery_from_result(result: Any) -> list[tuple[str, str]]:
    segments = getattr(result, "multimodal_segments", None) or []
    evidence: list[dict[str, Any]] = []
    for segment in segments:
        for caption in getattr(segment, "visual_captions", []) or []:
            if getattr(caption, "image_path", None):
                evidence.append(
                    {
                        "image_path": caption.image_path,
                        "frame_caption": caption.caption,
                        "timestamp": f"{caption.timestamp:.1f}s",
                        "evidence_type": "frame_caption",
                    }
                )
    return format_evidence_gallery_items(evidence)


def _gallery_from_evidence(evidence: list[dict]) -> list[str]:
    paths: list[str] = []
    for item in evidence or []:
        path = item.get("image_path")
        if path and path not in paths:
            paths.append(path)
    return paths


def _diagnostics(value: Any) -> dict[str, Any]:
    raw = _debug(value)
    summary = raw.get("summary") or {}
    storyline = raw.get("storyline") or {}
    unsupported = []
    missing = []
    for node in storyline.get("nodes", []) if isinstance(storyline, dict) else []:
        if node.get("status") in {"unsupported", "needs_review"}:
            unsupported.append(node.get("node_id"))
        if not node.get("evidence_segment_ids"):
            missing.append(node.get("node_id"))
    return {
        "warnings": summary.get("warnings", []),
        "unsupported_claims": unsupported,
        "evidence_missing": missing,
        "last_error": raw.get("error"),
        "rollback": "UI callbacks do not mutate current AppState when processing fails.",
    }


def _debug(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    return {}


def _stream_debug(value: Any) -> dict[str, Any]:
    return {
        "total": getattr(value, "total", 0),
        "events": _jsonable(getattr(value, "events", []))[-30:],
        "results": _jsonable(getattr(value, "results", [])),
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
