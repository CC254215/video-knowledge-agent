from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.models import FrameCorrelationScore, TranscriptSegment, VideoCorrelationProfile, VideoFrame
from app.runtime.scheduler import get_scheduler

CORRELATION_PROMPT = """
你是一名专业、严谨的视频内容相关性分析师。

我会给你来自同一视频的若干张截图描述，以及每张截图对应的台词/文本。

════════════════════════════════════
【第一步】视频类型判断（必须先执行）
════════════════════════════════════

在评分之前，请先综合所有截图，判断：
1. 这是什么类型的视频？
2. 该类型视频中，画面与台词的正常关系是什么？

类型参考与对应评分逻辑：

| 视频类型     | 画面与台词的预期关系                         |
|------------|---------------------------------------------|
| 教学/教程类  | 画面是教学场景，台词是知识讲解，无需字面对应   |
| 纪录片类    | 画面是事件/景象，台词是旁白解说，允许适度抽象  |
| 新闻类      | 画面须与报道内容强对应                        |
| 游戏解说类  | 游戏画面配技巧/剧情讲解，操作对应即可          |
| 美食/Vlog类 | 允许情感性旁白与画面松散关联                  |
| 健身类      | 动作画面与动作要领/计划讲解对应               |
| 广告类      | 画面与文案须高度匹配                          |

请在输出最开始写明：
视频类型：<判断结果>
评分依据：<该类型下画面与台词的关系说明>

════════════════════════════════════
【第二步】逐帧评分
════════════════════════════════════

基于第一步的判断，对每对【图片+文本】按以下规则打分：

评分区间定义：
· 1.0       图片与文本完全一致，语义与视觉一一对应
· 0.8～0.9  高度相关，主体/场景/主题一致
· 0.6～0.7  中等相关，属同一主题领域，画面与文字不完全对应
· 0.4～0.5  弱相关，有微弱联系，但主体/场景明显不符
· 0.2～0.3  几乎无关，仅存在极其间接的联系
· 0.0～0.1  完全无关，内容风马牛不相及

注意事项：
1. 过渡句处理："接下来我们看..."、"好了今天的内容到这里" 等
   → 场景合理则给 0.6～0.7，不得因无直接对应而判低分
2. 置信度低于 0.3 仅适用于：画面与台词属于完全不同主题领域，
   或存在明显剪辑错误/字幕错位
3. 必须基于视频整体类型判断，不得孤立评估单帧

════════════════════════════════════
【输出格式】（严格遵守，不得修改结构）
════════════════════════════════════

视频类型：<判断结果>
评分依据：<该类型下的评分逻辑说明>

输入图片：<图片内容描述>
输入文本：<对应台词内容>
场景合理性：<0.0～1.0，保留一位小数>
内容匹配度：<0.0～1.0，保留一位小数>
置信度：<0.0～1.0，保留一位小数>

（以上格式重复，直到所有帧评分完毕）

════════════════════════════════════
【示例】（严格参照格式）
════════════════════════════════════

───────────────────────────
示例一：纪录片类
───────────────────────────

视频类型：自然纪录片
评分依据：航拍/实景画面配旁白解说，画面与台词允许适度抽象，无需字面对应

输入图片：航拍的亚马逊雨林，大片绿色植被，河流蜿蜒穿过
输入文本：这片森林每年吸收约20亿吨二氧化碳，被称为地球之肺
场景合理性：0.9
内容匹配度：0.9
置信度：0.9
（纪录片旁白配实景画面，标准叙事方式，高度吻合）

输入图片：航拍的亚马逊雨林
输入文本：接下来我们去采访当地的原住民部落
场景合理性：0.8
内容匹配度：0.3
置信度：0.6
（过渡句，画面尚未切换，台词已转向下一段落，属正常剪辑节奏）

输入图片：航拍的亚马逊雨林
输入文本：这款跑鞋采用全新碳板缓震技术
场景合理性：0.0
内容匹配度：0.0
置信度：0.0
（雨林画面出现运动鞋广告台词，完全不属于同一视频内容）

───────────────────────────
示例二：新闻类
───────────────────────────

视频类型：电视新闻
评分依据：新闻画面须与播报内容强对应，记者出镜+现场背景是核心匹配点

输入图片：记者手持话筒站在洪水现场，背景可见被淹街道与救援人员
输入文本：目前受灾面积已超300平方公里，当地政府已启动一级应急响应
场景合理性：1.0
内容匹配度：0.9
置信度：0.9
（新闻现场画面与播报内容直接对应，强关联）

输入图片：记者站在洪水现场
输入文本：经济学家表示今年全球通胀压力依然较大
场景合理性：0.2
内容匹配度：0.0
置信度：0.1
（灾情现场突然切入经济话题，极不自然，疑似剪辑错误或字幕错位）

───────────────────────────
示例三：游戏解说类
───────────────────────────

视频类型：FPS游戏解说
评分依据：游戏实时画面配操作技巧/战术讲解，动作与解说对应即可

输入图片：第一人称射击游戏画面，玩家正在交火，屏幕右下角显示弹药数
输入文本：这把枪后坐力很大，压枪时要往左下方拉，节奏控制在三连发
场景合理性：0.9
内容匹配度：0.9
置信度：0.9
（游戏操作画面配技巧讲解，解说视频的典型模式）

输入图片：第一人称射击游戏画面，玩家正在激烈交火
输入文本：好了今天的视频到这里就结束了，喜欢的话记得点赞
场景合理性：0.6
内容匹配度：0.1
置信度：0.4
（台词是结尾话术，但画面仍在游戏中，可能是剪辑对不上，不应判为无关）

───────────────────────────
示例四：美食/Vlog类
───────────────────────────

视频类型：美食制作Vlog
评分依据：烹饪操作画面配步骤讲解或情感旁白，允许松散关联

输入图片：铁锅中食材翻炒，油烟升腾，食材颜色金黄
输入文本：大火翻炒两分钟，加入生抽和少许盐，出锅前淋上香油
场景合理性：1.0
内容匹配度：1.0
置信度：1.0
（烹饪画面与操作步骤台词完美对应）

输入图片：铁锅中食材翻炒
输入文本：这道菜是外婆教我的，每次吃都想起小时候在农村的夏天
场景合理性：0.9
内容匹配度：0.5
置信度：0.7
（Vlog常见情感旁白配烹饪画面，场景合理，内容属情感延伸而非直接对应）

输入图片：铁锅中食材翻炒
输入文本：撒哈拉沙漠年降水量不足25毫米
场景合理性：0.0
内容匹配度：0.0
置信度：0.0
（烹饪画面配沙漠地理知识，完全错误匹配）

───────────────────────────
示例五：健身类
───────────────────────────

视频类型：健身教程
评分依据：动作演示画面配动作要领讲解，动作与指导须直接对应

输入图片：男性在健身房做深蹲，动作标准，背景有器械
输入文本：注意膝盖不要超过脚尖，核心收紧，下蹲至大腿平行于地面
场景合理性：1.0
内容匹配度：0.9
置信度：0.9
（深蹲动作画面与深蹲要领讲解直接对应）

输入图片：男性在健身房做深蹲
输入文本：今天我们聊聊如何制定增肌饮食计划
场景合理性：0.7
内容匹配度：0.2
置信度：0.5
（同属健身主题，但画面是训练动作，台词转向饮食规划，存在跨段落跳跃）

输入图片：男性在健身房做深蹲
输入文本：本片带您走进神秘的马里亚纳海沟
场景合理性：0.0
内容匹配度：0.0
置信度：0.0
（健身画面配海洋纪录片台词，完全不属于同一视频）
"""


def assess_video_text_correlation(
    video_id: str,
    transcript_segments: list[TranscriptSegment],
    video_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> VideoCorrelationProfile:
    settings = settings or get_settings()
    text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
    if len(text) < settings.correlation_text_min_chars:
        return _save(_profile_from_confidence([], 1.0, "high", "text_too_short", "text below threshold; default high video relevance"), output_dir)
    if not video_path or not Path(video_path).exists():
        return _save(_profile_from_confidence([], 1.0, "text_only", "video_unavailable", "video unavailable; skipped visual analysis"), output_dir)
    if not settings.has_correlation_config:
        return _save(_profile_from_confidence([], 1.0, "high", "missing_model", "correlation model config missing; preserve default multimodal behavior"), output_dir)

    samples = sample_random_correlation_frames(video_id, video_path, output_dir or Path(video_path).parent, max_frames=settings.correlation_sample_frames)
    pairs = [(frame, _text_for_timestamp(frame.timestamp, transcript_segments)) for frame in samples]
    pairs = [(frame, text) for frame, text in pairs if text.strip()]
    if not pairs:
        return _save(_profile_from_confidence([], 1.0, "text_only", "video_unavailable", "no frame/text pairs available"), output_dir)
    try:
        scored = _score_frame_text_batch(pairs, settings)
    except Exception as exc:  # noqa: BLE001
        scored = [
            FrameCorrelationScore(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                text=text,
                confidence=0.5,
                reason=f"batch_scoring_failed: {str(exc).splitlines()[0][:300]}",
            )
            for frame, text in pairs
        ]
    confidences = [item.confidence for item in scored]
    avg = sum(confidences) / len(confidences)
    profile = _profile_from_confidence(confidences, avg, _level(avg, settings), "success", "LLM frame/text correlation completed")
    profile.frame_scores = scored
    return _save(profile, output_dir)


def sample_random_correlation_frames(video_id: str, video_path: str | Path, output_dir: str | Path, max_frames: int = 10) -> list[VideoFrame]:
    path = Path(video_path)
    duration = _probe_duration(path)
    if duration <= 0:
        return []
    rng = random.Random(video_id)
    timestamps = sorted(rng.uniform(0.0, max(0.1, duration)) for _ in range(max(1, max_frames)))
    frame_dir = Path(output_dir) / "frames" / "correlation_probe"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("corr_*.jpg"):
        old.unlink(missing_ok=True)
    frames: list[VideoFrame] = []
    fallback_source: Path | None = None
    for index, timestamp in enumerate(timestamps):
        target = frame_dir / f"corr_{index:04d}.jpg"
        source = fallback_source or path
        command = _single_frame_command(source, target, timestamp)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 and fallback_source is None and _looks_like_decode_failure(result.stderr):
            fallback_source = _transcode_to_h264(path, Path(output_dir))
            if fallback_source:
                result = subprocess.run(_single_frame_command(fallback_source, target, timestamp), capture_output=True, text=True, check=False)
        if result.returncode == 0 and target.exists():
            frames.append(VideoFrame(frame_id=f"corr_{index:04d}", timestamp=timestamp, path=str(target), selected=True, selection_reason="correlation_probe"))
    if not frames:
        try:
            from app.vision.frame_extractor import extract_frames

            extracted = extract_frames(video_id, path, output_dir, fps=max(0.001, max_frames / duration), start=0.0, end=duration)
            rng.shuffle(extracted)
            frames = sorted(extracted[:max_frames], key=lambda frame: frame.timestamp)
            for frame in frames:
                frame.selected = True
                frame.selection_reason = "correlation_probe_fps_fallback"
        except Exception:
            return []
    return frames


def _single_frame_command(source: Path, target: Path, timestamp: float) -> list[str]:
    return [
        _ffmpeg_executable(),
        "-y",
        "-hwaccel",
        "none",
        "-analyzeduration",
        "100M",
        "-probesize",
        "100M",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(target),
    ]


def _looks_like_decode_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        signal in lowered
        for signal in (
            "av1",
            "missing sequence header",
            "failed to get pixel format",
            "function not implemented",
            "error while decoding stream",
        )
    )


def _transcode_to_h264(source: Path, output_dir: Path) -> Path | None:
    fallback_path = output_dir / "downloads" / "source_video_h264_fallback.mp4"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    if fallback_path.exists() and fallback_path.stat().st_size > 0:
        return fallback_path
    command = [
        _ffmpeg_executable(),
        "-y",
        "-hwaccel",
        "none",
        "-analyzeduration",
        "100M",
        "-probesize",
        "100M",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(fallback_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return fallback_path if result.returncode == 0 and fallback_path.exists() else None


def _score_frame_text_batch(pairs: list[tuple[VideoFrame, str]], settings: Settings) -> list[FrameCorrelationScore]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _batch_prompt(pairs)}]
    for index, (frame, _text) in enumerate(pairs, start=1):
        content.append({"type": "text", "text": f"第{index}组图片如下。"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(frame.path))}})
    payload = {"model": settings.runtime_correlation_model, "messages": [{"role": "user", "content": content}], "temperature": 0.1}

    def _request() -> httpx.Response:
        result = httpx.post(
            _join_openai_endpoint(settings.runtime_correlation_base_url, "chat/completions"),
            headers={"Authorization": f"Bearer {settings.llm_correlation_api_key}"},
            json=payload,
            timeout=settings.request_timeout_seconds,
        )
        if result.status_code == 429 or result.status_code >= 500:
            raise RuntimeError(f"retryable_http_status={result.status_code}, body={result.text[:1000]}")
        return result

    response = get_scheduler(settings).call("vlm", _request)
    if response.status_code >= 400:
        raise RuntimeError(f"correlation request failed: status={response.status_code}, body={response.text[:1000]}")
    output = response.json()["choices"][0]["message"]["content"]
    confidences = _parse_confidences(output)
    return [
        FrameCorrelationScore(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            text=text,
            confidence=confidences[index] if index < len(confidences) else 0.5,
            reason=_reason_for_index(output, index),
        )
        for index, (frame, text) in enumerate(pairs)
    ]


def _batch_prompt(pairs: list[tuple[VideoFrame, str]]) -> str:
    rows = []
    for index, (frame, text) in enumerate(pairs, start=1):
        rows.append(f"【第{index}组】\nframe_id: {frame.frame_id}\ntimestamp: {frame.timestamp:.2f}s\n输入文本：{text[:900]}")
    return CORRELATION_PROMPT + "\n\n请按下面这些图片与文本对逐一评分：\n\n" + "\n\n".join(rows)


def _parse_confidences(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"置信度\s*[：:]\s*([01](?:\.\d+)?)", text):
        values.append(max(0.0, min(1.0, float(match.group(1)))))
    return values


def _reason_for_index(text: str, index: int) -> str:
    blocks = re.split(r"输入图片\s*[：:]", text)
    if index + 1 < len(blocks):
        return blocks[index + 1].strip()[:500]
    return "parsed_from_batch_correlation_prompt"


def _profile_from_confidence(confidences: list[float], avg: float, level: str, status: str, reason: str) -> VideoCorrelationProfile:
    if level == "low":
        return VideoCorrelationProfile(frame_confidences=confidences, avg_confidence=avg, frame_interval=60.0, refine_frame_count=2, representative_frame_strategy="random", relevance_level="low", status=status, reason=reason)  # type: ignore[arg-type]
    if level == "medium":
        return VideoCorrelationProfile(frame_confidences=confidences, avg_confidence=avg, frame_interval=2.0, refine_frame_count=5, representative_frame_strategy="default", relevance_level="medium", status=status, reason=reason)  # type: ignore[arg-type]
    return VideoCorrelationProfile(frame_confidences=confidences, avg_confidence=avg, frame_interval=1.0, refine_frame_count=10, representative_frame_strategy="default", relevance_level=level if level == "text_only" else "high", status=status, reason=reason)  # type: ignore[arg-type]


def _level(avg: float, settings: Settings) -> str:
    if avg < settings.correlation_low_threshold:
        return "low"
    if avg < settings.correlation_high_threshold:
        return "medium"
    return "high"


def _text_for_timestamp(timestamp: float, segments: list[TranscriptSegment]) -> str:
    for segment in segments:
        if segment.start <= timestamp <= segment.end and segment.text.strip():
            return segment.text.strip()
    if not segments:
        return ""
    nearest = min(segments, key=lambda segment: min(abs(timestamp - segment.start), abs(timestamp - segment.end)))
    return nearest.text.strip()


def _save(profile: VideoCorrelationProfile, output_dir: str | Path | None) -> VideoCorrelationProfile:
    if output_dir:
        (Path(output_dir) / "video_correlation.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _probe_duration(path: Path) -> float:
    result = subprocess.run([_ffprobe_executable(), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return 0.0
    try:
        return float((json.loads(result.stdout).get("format") or {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def _ffmpeg_executable() -> str:
    local = Path("tools/ffmpeg/bin/ffmpeg.exe").resolve()
    if local.exists():
        return str(local)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg is required for correlation frame sampling.")


def _ffprobe_executable() -> str:
    local = Path("tools/ffmpeg/bin/ffprobe.exe").resolve()
    if local.exists():
        return str(local)
    exe = shutil.which("ffprobe")
    return exe or _ffmpeg_executable().replace("ffmpeg", "ffprobe")


def _join_openai_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(endpoint):
        return base
    return f"{base}/{endpoint}"
