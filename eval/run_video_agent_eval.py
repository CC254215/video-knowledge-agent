"""Small, fixed EgoSchema evaluation of the current video pipeline.

Run from the repository root:
    python eval/run_video_agent_eval.py --max-samples 3
    python eval/run_video_agent_eval.py --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import pipeline  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ingestion.local_file import metadata_from_local_file  # noqa: E402
from app.models import TranscriptSegment, VideoTranscript  # noqa: E402


OFFICIAL = "https://raw.githubusercontent.com/egoschema/EgoSchema/main"
EVAL_DIR = ROOT / "eval"
DATA_DIR = ROOT / "data" / "eval" / "egoschema"
IDS_PATH = EVAL_DIR / "egoschema_20_ids.json"
RESULTS_PATH = EVAL_DIR / "results.json"
REPORT_PATH = EVAL_DIR / "report.md"
LETTERS = "ABCDE"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def official_json(name: str) -> Any:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(f"{OFFICIAL}/{name}", follow_redirects=True, timeout=60)
        response.raise_for_status()
        atomic_json(path, response.json())
    return json.loads(path.read_text(encoding="utf-8"))


def selected_ids(answers: dict[str, int]) -> list[str]:
    if IDS_PATH.exists():
        ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    else:
        ids = random.Random(42).sample(sorted(answers), 20)
        atomic_json(IDS_PATH, ids)
    if len(ids) != 20 or len(set(ids)) != 20 or any(uid not in answers for uid in ids):
        raise ValueError("egoschema_20_ids.json must contain 20 unique public-subset IDs")
    return ids


def download_video(question: dict[str, Any]) -> Path:
    uid = question["q_uid"]
    path = DATA_DIR / "videos" / f"{uid}.mp4"
    if path.is_file() and path.stat().st_size > 1024:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://drive.usercontent.google.com/download"
           f"?id={question['google_drive_id']}&export=download&confirm=t")
    last_error: Exception | None = None
    for attempt in range(2):
        temp = path.with_suffix(".mp4.part")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                response.raise_for_status()
                if "video/" not in response.headers.get("content-type", ""):
                    raise ValueError("Google Drive did not return a video; access may be restricted")
                expected = int(response.headers.get("content-length", "0"))
                with temp.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            if temp.stat().st_size <= 1024 or (expected and temp.stat().st_size != expected):
                raise ValueError("Incomplete video download")
            temp.replace(path)
            return path
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt == 0:
                time.sleep(2)
    raise RuntimeError(f"Video download failed for {uid}: {last_error}")


def parse_answer(answer: str) -> str | None:
    text = answer.strip()
    if re.search(r"\b[A-E]\s*(?:or|/|或)\s*[A-E]\b", text, flags=re.IGNORECASE):
        return None
    if re.fullmatch(r"[A-Ea-e][.)]?", text):
        return text[0].upper()
    patterns = (
        r"(?:final\s+answer|answer|i\s+choose|my\s+choice)\s*(?:is|:)?\s*([A-E])\b",
        r"^\s*([A-E])[.)]\s+",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        distinct = {match.upper() for match in matches}
        if len(distinct) == 1:
            return distinct.pop()
        if len(distinct) > 1:
            return None
    return None


class UsageRecorder:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_calls = 0
        self.calls_without_usage = 0
        self.frames_viewed = 0

    def record(self, response: httpx.Response, request_kwargs: dict[str, Any]) -> None:
        if response.status_code >= 400:
            return
        payload = request_kwargs.get("json")
        if not isinstance(payload, dict) or "model" not in payload:
            return
        self.model_calls += 1
        self.frames_viewed += _count_images(payload.get("messages"))
        try:
            usage = response.json().get("usage")
        except (ValueError, AttributeError):
            usage = None
        if not isinstance(usage, dict):
            self.calls_without_usage += 1
            return
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if not isinstance(prompt, int) or not isinstance(completion, int):
            self.calls_without_usage += 1
            return
        self.input_tokens += prompt
        self.output_tokens += completion

    def metrics(self) -> dict[str, int | None]:
        complete = self.model_calls > 0 and self.calls_without_usage == 0
        return {
            "input_tokens": self.input_tokens if complete else None,
            "output_tokens": self.output_tokens if complete else None,
            "frames_viewed": self.frames_viewed if self.model_calls else None,
        }


def _count_images(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(
        1
        for message in messages if isinstance(message, dict)
        for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


@contextmanager
def record_model_usage(recorder: UsageRecorder):
    original = httpx.post

    def measured_post(*args: Any, **kwargs: Any) -> httpx.Response:
        response = original(*args, **kwargs)
        recorder.record(response, kwargs)
        return response

    httpx.post = measured_post
    try:
        yield
    finally:
        httpx.post = original


def load_results() -> list[dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return []
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("results.json must contain a list")
    return rows


def add_review_context(row: dict[str, Any], question: dict[str, Any]) -> bool:
    """Attach official question and local evidence paths without a model call."""
    uid = row["id"]
    video = DATA_DIR / "videos" / f"{uid}.mp4"
    context: dict[str, Any] = {
        "question": question["question"],
        "choices": {LETTERS[i]: question[f"option {i}"] for i in range(5)},
        "video_path": str(video.relative_to(ROOT)),
    }
    if video.exists():
        processed = pipeline.video_dir(metadata_from_local_file(video).video_id)
        context["frame_captions_path"] = str((processed / "frame_captions.json").relative_to(ROOT))
        traces = list((processed / "qa_traces").glob("*.json"))
        if traces:
            context["trace_path"] = str(max(traces, key=lambda path: path.stat().st_mtime).relative_to(ROOT))
    changed = any(row.get(key) != value for key, value in context.items())
    row.update(context)
    return changed


def run_case(question: dict[str, Any], truth: int, video_path: Path) -> dict[str, Any]:
    uid = question["q_uid"]
    prompt = "\n".join([
        question["question"],
        "Choices:",
        *(f"{LETTERS[index]}. {question[f'option {index}']}" for index in range(5)),
        "Choose the best answer. End with `Answer: A`, `Answer: B`, `Answer: C`, `Answer: D`, or `Answer: E`.",
    ])
    settings = get_settings().model_copy(update={
        "api_max_retries": 1,
        "vlm_caption_retries": 1,
        "pipeline_stage_restart_limit": 0,
    })
    usage = UsageRecorder()
    started = time.monotonic()
    row: dict[str, Any] = {
        "id": uid,
        "prediction": None,
        "ground_truth": LETTERS[int(truth)],
        "correct": False,
        "latency": None,
        "agent_steps": None,
        "tool_calls": None,
        "input_tokens": None,
        "output_tokens": None,
        "frames_viewed": None,
    }
    try:
        with record_model_usage(usage):
            prepare_silent_video(video_path, settings)
            processed = pipeline.process(file_path=video_path, settings=settings)
            turn = pipeline.chat(
                processed["video_id"], prompt, settings=settings,
                persist=False, allow_refinement=True, persist_trace=True,
                persist_refinement=False,
            )
        row["prediction"] = parse_answer(turn.answer)
        row["correct"] = row["prediction"] == row["ground_truth"]
        row["status"] = "completed" if row["prediction"] else "invalid"
        row["answer"] = turn.answer
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        row["latency"] = round(time.monotonic() - started, 3)
        row.update(usage.metrics())
        row["model_calls"] = usage.model_calls
        row["calls_without_usage"] = usage.calls_without_usage
    return row


def prepare_silent_video(video_path: Path, settings: Any) -> None:
    """Supply a neutral time anchor for clips with no audio stream.

    The existing segment builder requires a nonempty transcript. This adapter
    records the absence of speech without inventing video content.
    """
    import av

    with av.open(str(video_path)) as container:
        if container.streams.audio:
            return
        duration = (container.duration or 0) / 1_000_000
    metadata = metadata_from_local_file(video_path)
    metadata.duration = duration
    out_dir = pipeline.video_dir(metadata.video_id, settings)
    metadata_path = out_dir / "metadata.json"
    transcript_path = out_dir / "transcript.json"
    if metadata_path.exists() and transcript_path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor = TranscriptSegment(
        segment_id="silent_video_0000",
        start=0.0,
        end=max(duration, 0.001),
        text="Silent video. No spoken transcript is available; use visual evidence.",
    )
    pipeline.write_json(metadata_path, metadata)
    pipeline.write_json(transcript_path, VideoTranscript(video_id=metadata.video_id, segments=[anchor], source="unknown"))


def _average(rows: list[dict[str, Any]], key: str) -> str:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return f"{statistics.mean(values):,.1f}" if len(values) == len(rows) and values else "未记录"


def render_report(rows: list[dict[str, Any]], target: int) -> str:
    correct = sum(row.get("correct") is True for row in rows)
    wrong = sum(row.get("status") == "completed" and not row.get("correct") for row in rows)
    invalid = sum(row.get("status") == "invalid" for row in rows)
    failed = sum(row.get("status") == "failed" for row in rows)
    latencies = [row["latency"] for row in rows if isinstance(row.get("latency"), (int, float))]
    tokens = [row["input_tokens"] + row["output_tokens"] for row in rows
              if isinstance(row.get("input_tokens"), int) and isinstance(row.get("output_tokens"), int)]
    tokens_complete = len(tokens) == len(rows) and bool(rows)
    total_tokens = sum(tokens) if tokens_complete else None
    input_total = sum(row["input_tokens"] for row in rows) if tokens_complete else None
    output_total = sum(row["output_tokens"] for row in rows) if tokens_complete else None

    def ids(predicate: Any) -> str:
        return "、".join(row["id"] for row in rows if predicate(row)) or "无"

    lines = [
        "# 视频 Agent 轻量评测报告", "",
        "**数据集：**EgoSchema 官方公开子集；随机种子 42，固定抽取 20 题。", "",
        "## 结果概览", "",
        "| 指标 | 结果 |", "| --- | ---: |",
        f"| 已完成题数 | {len(rows)} / {target} |",
        f"| 答对 / 答错 / 未给出选项 / 运行失败 | {correct} / {wrong} / {invalid} / {failed} |",
        f"| 准确率（答对数 / 已完成题数） | {correct / len(rows):.1%} |" if rows else "| 准确率 | 未记录 |",
        f"| 平均每题耗时 | {statistics.mean(latencies):.1f} 秒 |" if latencies else "| 平均每题耗时 | 未记录 |",
        f"| 耗时中位数 | {statistics.median(latencies):.1f} 秒 |" if latencies else "| 耗时中位数 | 未记录 |",
        f"| 平均 Agent 步数 | {_average(rows, 'agent_steps')} |",
        f"| 平均工具调用次数 | {_average(rows, 'tool_calls')} |",
        f"| 平均送入视觉模型的帧数 | {_average(rows, 'frames_viewed')} |",
        f"| 输入 Token 总数 | {input_total:,} |" if input_total is not None else "| 输入 Token 总数 | 未记录（部分模型调用缺少用量） |",
        f"| 输出 Token 总数 | {output_total:,} |" if output_total is not None else "| 输出 Token 总数 | 未记录（部分模型调用缺少用量） |",
        f"| Token 总数 | {total_tokens:,} |" if total_tokens is not None else "| Token 总数 | 未记录 |",
        f"| 平均每题 Token | {total_tokens / len(rows):,.0f} |" if total_tokens is not None else "| 平均每题 Token | 未记录 |",
        f"| 每答对一题消耗的 Token | {total_tokens / correct:,.0f} |" if total_tokens is not None and correct else "| 每答对一题消耗的 Token | 未记录 |",
        f"| 每 10 万 Token 答对题数 | {correct / total_tokens * 100000:.2f} |" if total_tokens else "| 每 10 万 Token 答对题数 | 未记录 |",
        "",
        "每 10 万 Token 答对题数是**自定义工程指标**，并非 EgoSchema 官方指标。耗时和 Token 包含视频预处理及问答，不包含下载。帧数按经 HTTP 发送给视觉模型的图像输入统计；当前无法直接取得的 Agent 步数和工具调用次数标为“未记录”。", "",
        "## 结果解读", "",
        f"本轮答对 {correct} 题，明确给出 A–E 但答错 {wrong} 题，未给出可解析选项 {invalid} 题。未给出选项按错误计入准确率。",
        "这些官方视频没有音轨；评测适配器为现有管线建立了一个不含视频事实的“无语音”时间锚点。该锚点进入 transcript_text，现有系统会把它标成 speech 证据，并可能判为文本主导。因此，本轮分数受到临时适配方式影响，不能视作干净的纯视觉能力测量。",
        "当前 Agent 主要依据抽帧后的静态视觉描述答题，涉及连续动作、先后顺序与关键转折的题目容易缺少足够证据。部分题虽触发补帧，最终仍选择弃答。现有视觉问题关键词以中文为主，本轮英文题未触发该关键词规则。评测提示词中的 Choices: 和 Answer: 还触发了冒号时间导航规则，导致全部 20 题被归为 time_navigation、要求 speech 证据；这是评测入口偏差，修复后应重新测量。",
        "这是当前整套视频处理与问答系统在固定 20 题上的体检结果，不应当作大样本官方排行榜成绩。逐题原文、回答和证据路径见下文。", "",
        "## 错误分析", "",
        f"- 答对题目：{ids(lambda row: row.get('correct') is True)}",
        f"- 给出选项但答错：{ids(lambda row: row.get('status') == 'completed' and not row.get('correct'))}",
        f"- 未给出有效选项：{ids(lambda row: row.get('status') == 'invalid')}",
        f"- 运行失败：{ids(lambda row: row.get('status') == 'failed')}",
    ]
    rankings = (
        ("Token 消耗最高", lambda row: (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0), lambda row: row.get("input_tokens") is not None and row.get("output_tokens") is not None, "Token"),
        ("耗时最长", lambda row: row.get("latency") or 0, lambda row: row.get("latency") is not None, "秒"),
        ("工具调用最多", lambda row: row.get("tool_calls") or 0, lambda row: row.get("tool_calls") is not None, "次"),
    )
    for title, metric, available, unit in rankings:
        lines += ["", f"## {title}的 3 题", ""]
        eligible = [row for row in rows if available(row)]
        lines += [f"- {row['id']}：{metric(row):,.1f} {unit}；{'答对' if row.get('correct') else '未答对'}"
                  for row in sorted(eligible, key=metric, reverse=True)[:3]] or ["未记录"]
    lines += ["", "## 逐题查看", "",
              "题目和选项保留官方英文原文，Agent 回答保留原始输出；其余说明与统计均为中文。视频、视觉描述和推理 trace 均在本地。", ""]
    for index, row in enumerate(rows, start=1):
        choices = row.get("choices") or {}
        def local_link(label: str, key: str) -> str:
            path = row.get(key)
            if not path:
                return "未记录"
            return f"[{label}](../{path.replace(chr(92), '/')}) (`{path}`)"

        token_count = ((row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
                       if row.get("input_tokens") is not None and row.get("output_tokens") is not None else "未记录")
        status = {"completed": "已给出选项", "invalid": "未给出有效选项", "failed": "运行失败"}.get(row.get("status"), "未知")
        lines += [
            f"### {index:02d}. {row['id']}", "",
            f"**官方题目：**{row.get('question', '未记录')}", "",
            *(f"- {letter}. {choices[letter]}" for letter in LETTERS if letter in choices),
            "",
            f"**结果：**{status} · Agent 选择 {row.get('prediction') or '无'} · "
            f"标准答案 {row.get('ground_truth')} · 消耗 {token_count} Token", "",
            f"**视频：**{local_link('打开 MP4', 'video_path')}  ",
            f"**视觉描述：**{local_link('打开帧描述', 'frame_captions_path')}  ",
            f"**推理记录：**{local_link('打开 trace', 'trace_path')}", "",
            "**Agent 原始回答：**", "",
            *(f"> {part}" for part in (row.get("answer") or row.get("error") or "未记录").splitlines()),
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--download-only", action="store_true", help="Fetch selected videos without running the agent")
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly rerun failed items")
    args = parser.parse_args()
    if not 1 <= args.max_samples <= 20:
        parser.error("--max-samples must be between 1 and 20")

    questions = {row["q_uid"]: row for row in official_json("questions.json")}
    answers = official_json("subset_answers.json")
    ids = selected_ids(answers)[:args.max_samples]
    results = load_results()
    by_id = {row["id"]: row for row in results}
    changed = False
    for row in results:
        if row["id"] in questions:
            changed = add_review_context(row, questions[row["id"]]) or changed
    if changed:
        atomic_json(RESULTS_PATH, results)
    for index, uid in enumerate(ids, start=1):
        if uid in by_id and not (args.retry_failed and by_id[uid].get("status") == "failed"):
            print(f"[{index}/{len(ids)}] skip {uid} (saved)", flush=True)
            continue
        print(f"[{index}/{len(ids)}] downloading {uid}", flush=True)
        try:
            video = download_video(questions[uid])
            if args.download_only:
                print(f"  ready: {video.stat().st_size / 1024**2:.1f} MiB", flush=True)
                continue
            print(f"  running agent: {video.stat().st_size / 1024**2:.1f} MiB", flush=True)
            row = run_case(questions[uid], answers[uid], video)
        except Exception as exc:  # noqa: BLE001
            row = {"id": uid, "prediction": None, "ground_truth": LETTERS[int(answers[uid])],
                   "correct": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:1000],
                   "latency": None, "agent_steps": None, "tool_calls": None,
                   "input_tokens": None, "output_tokens": None, "frames_viewed": None}
        by_id[uid] = row
        add_review_context(row, questions[uid])
        results = [by_id[key] for key in selected_ids(answers) if key in by_id]
        atomic_json(RESULTS_PATH, results)
        scoped_results = [by_id[key] for key in ids if key in by_id]
        REPORT_PATH.write_text(render_report(scoped_results, args.max_samples), encoding="utf-8")
        print(f"  {row['status']}: prediction={row.get('prediction')} truth={row['ground_truth']} latency={row.get('latency')}", flush=True)
    if not args.download_only:
        scoped_results = [by_id[key] for key in ids if key in by_id]
        REPORT_PATH.write_text(render_report(scoped_results, args.max_samples), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
