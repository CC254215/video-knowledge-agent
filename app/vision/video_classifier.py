from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean

from app.config import Settings, get_settings
from app.models import VideoVisualProfile


def classify_video_visual_intensity(
    video_id: str,
    video_path: str | Path,
    output_dir: str | Path,
    has_subtitles: bool = False,
    settings: Settings | None = None,
) -> VideoVisualProfile:
    """Lightweight visual gate before expensive multimodal processing.

    This stage never calls a vision model. It samples a small number of scaled
    frames and computes cheap frame-diff, edge-density, color-variation, and
    metadata features. The resulting profile controls downstream frame density
    while keeping MultimodalSegment output schema unchanged.
    """
    settings = settings or get_settings()
    path = Path(video_path)
    if not settings.visual_classifier_enabled or not path.exists():
        return _default_profile(video_id, path, has_subtitles, settings, reason="classifier_disabled_or_missing_video")

    width, height, fps = _probe_video(path)
    sample_dir = Path(output_dir) / "frames" / "visual_probe"
    samples = _sample_probe_frames(path, sample_dir)
    motion_score, complexity_score, color_score = _compute_lightweight_scores(samples)
    resolution_score = _resolution_score(width, height)
    subtitle_discount = 0.08 if has_subtitles else 0.0
    visual_strength = _clamp(
        0.36 * motion_score
        + 0.32 * complexity_score
        + 0.18 * color_score
        + 0.14 * resolution_score
        - subtitle_discount
    )
    visual_class = "strong_visual" if visual_strength >= settings.visual_strength_threshold else "weak_visual"
    profile = VideoVisualProfile(
        video_id=video_id,
        video_path=str(path),
        motion_score=motion_score,
        visual_complexity_score=complexity_score,
        color_variation_score=color_score,
        resolution_width=width,
        resolution_height=height,
        fps=fps,
        has_subtitles=has_subtitles,
        visual_strength_score=visual_strength,
        visual_class=visual_class,
        recommended_frame_fps=settings.strong_visual_frame_fps if visual_class == "strong_visual" else settings.weak_visual_frame_fps,
        recommended_max_frames_per_segment=settings.strong_visual_max_frames_per_segment if visual_class == "strong_visual" else settings.weak_visual_max_frames_per_segment,
        recommended_min_representative_frames=12 if visual_class == "strong_visual" else 3,
        reason=(
            f"motion={motion_score:.2f}, complexity={complexity_score:.2f}, color={color_score:.2f}, "
            f"resolution={resolution_score:.2f}, subtitles={has_subtitles}"
        ),
    )
    save_visual_profile(profile, output_dir)
    return profile


def save_visual_profile(profile: VideoVisualProfile, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "visual_profile.json"
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_visual_profile(output_dir: str | Path) -> VideoVisualProfile | None:
    path = Path(output_dir) / "visual_profile.json"
    if not path.exists():
        return None
    return VideoVisualProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _default_profile(video_id: str, path: Path, has_subtitles: bool, settings: Settings, reason: str) -> VideoVisualProfile:
    profile = VideoVisualProfile(
        video_id=video_id,
        video_path=str(path),
        motion_score=0.0,
        visual_complexity_score=0.0,
        color_variation_score=0.0,
        has_subtitles=has_subtitles,
        visual_strength_score=0.0,
        visual_class="weak_visual",
        recommended_frame_fps=settings.weak_visual_frame_fps,
        recommended_max_frames_per_segment=settings.weak_visual_max_frames_per_segment,
        recommended_min_representative_frames=3,
        reason=reason,
    )
    return profile


def _sample_probe_frames(video_path: Path, output_dir: Path, max_frames: int = 12) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("probe_*.jpg"):
        old.unlink(missing_ok=True)
    pattern = output_dir / "probe_%04d.jpg"
    command = [
        _ffmpeg_executable(),
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=0.25,scale=160:-1",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "5",
        str(pattern),
    ]
    subprocess.run(command, capture_output=True, text=True, check=False)
    return sorted(output_dir.glob("probe_*.jpg"))[:max_frames]


def _compute_lightweight_scores(paths: list[Path]) -> tuple[float, float, float]:
    if len(paths) < 2:
        return 0.0, 0.0, 0.0
    try:
        from PIL import Image, ImageFilter, ImageStat
    except Exception:
        return _byte_diff_score(paths), 0.0, 0.0

    grayscale = [Image.open(path).convert("L").resize((96, 54)) for path in paths]
    rgb_images = [Image.open(path).convert("RGB").resize((64, 36)) for path in paths]
    motion_values: list[float] = []
    for left, right in zip(grayscale, grayscale[1:], strict=False):
        left_pixels = list(left.getdata())
        right_pixels = list(right.getdata())
        diff = mean(abs(a - b) for a, b in zip(left_pixels, right_pixels, strict=False)) / 255.0
        motion_values.append(diff)
    edge_values = []
    for image in grayscale:
        edges = image.filter(ImageFilter.FIND_EDGES)
        edge_values.append(mean(edges.getdata()) / 255.0)
    color_values = []
    for image in rgb_images:
        stat = ImageStat.Stat(image)
        color_values.append(mean(stat.stddev) / 128.0)
    return _clamp(mean(motion_values) * 2.8), _clamp(mean(edge_values) * 3.0), _clamp(mean(color_values))


def _byte_diff_score(paths: list[Path]) -> float:
    values: list[float] = []
    for left, right in zip(paths, paths[1:], strict=False):
        left_bytes = left.read_bytes()[:4096]
        right_bytes = right.read_bytes()[:4096]
        if not left_bytes or not right_bytes:
            continue
        size = min(len(left_bytes), len(right_bytes))
        values.append(sum(1 for index in range(size) if left_bytes[index] != right_bytes[index]) / size)
    return _clamp(mean(values) if values else 0.0)


def _probe_video(path: Path) -> tuple[int | None, int | None, float | None]:
    ffprobe = _ffprobe_executable()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, None, None
    try:
        stream = (json.loads(result.stdout).get("streams") or [{}])[0]
        return int(stream.get("width")), int(stream.get("height")), _parse_rate(str(stream.get("r_frame_rate") or "0/1"))
    except Exception:
        return None, None, None


def _resolution_score(width: int | None, height: int | None) -> float:
    if not width or not height:
        return 0.0
    pixels = width * height
    return _clamp(math.log(max(pixels, 1), 1920 * 1080) / 1.15)


def _parse_rate(value: str) -> float | None:
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return None
    top, bottom = value.split("/", 1)
    try:
        denominator = float(bottom)
        return float(top) / denominator if denominator else None
    except ValueError:
        return None


def _ffmpeg_executable() -> str:
    local = Path("tools/ffmpeg/bin/ffmpeg.exe").resolve()
    if local.exists():
        return str(local)
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    raise RuntimeError("ffmpeg is required for lightweight visual classification.")


def _ffprobe_executable() -> str:
    local = Path("tools/ffmpeg/bin/ffprobe.exe").resolve()
    if local.exists():
        return str(local)
    exe = shutil.which("ffprobe")
    return exe or _ffmpeg_executable().replace("ffmpeg", "ffprobe")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
