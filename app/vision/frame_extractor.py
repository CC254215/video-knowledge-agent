from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from app.models import VideoFrame


def extract_frames(
    video_id: str,
    video_path: str | Path,
    video_dir: str | Path,
    fps: float = 1.0,
    start: float | None = None,
    end: float | None = None,
) -> list[VideoFrame]:
    """Extract a candidate frame pool with ffmpeg.

    The default 1 FPS output is only a candidate pool. Downstream dedupe/keyframe selection decides
    which frames are worth VLM captioning.
    """
    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(f"Video file does not exist: {source}")
    output_dir = Path(video_dir) / "frames" / ("refined" if start is not None else "raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    if start is not None:
        for old in output_dir.glob("frame_*.jpg"):
            old.unlink(missing_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    errors: list[str] = []
    for ffmpeg in _ffmpeg_executables():
        command = _frame_extract_command(ffmpeg, source, pattern, fps, start, end)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            break
        errors.append(f"[{ffmpeg}] {result.stderr.strip()}")
        fallback_source = _transcode_to_h264_if_needed(source, Path(video_dir), ffmpeg, result.stderr)
        if fallback_source is None:
            continue
        fallback_command = _frame_extract_command(ffmpeg, fallback_source, pattern, fps, start, end)
        fallback_result = subprocess.run(fallback_command, capture_output=True, text=True, check=False)
        if fallback_result.returncode == 0:
            break
        errors.append(f"[{ffmpeg}][h264-fallback] {fallback_result.stderr.strip()}")
    else:
        combined = " | ".join(errors)
        lowered = combined.lower()
        if "av1" in lowered:
            raise RuntimeError(
                "ffmpeg frame extraction failed across all decoders (AV1). "
                "Install or use an ffmpeg build with AV1 software decoding support "
                "(e.g. libdav1d/libaom), then retry. Details: "
                + combined
            )
        raise RuntimeError("ffmpeg frame extraction failed across all decoders: " + combined)
    frames = frames_from_directory(output_dir, fps=fps, start=start or 0.0, video_id=video_id)
    (Path(video_dir) / "frames.json").write_text(
        json.dumps([frame.model_dump() for frame in frames], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frames


def _transcode_to_h264_if_needed(
    source: Path,
    video_dir: Path,
    ffmpeg: str,
    stderr: str,
) -> Path | None:
    """Try one software-decode fallback for AV1-like decode failures."""
    lowered = stderr.lower()
    decode_signals = (
        "av1",
        "missing sequence header",
        "failed to get pixel format",
        "function not implemented",
        "error while decoding stream",
    )
    if not any(signal in lowered for signal in decode_signals):
        return None
    fallback_path = video_dir / "downloads" / "source_video_h264_fallback.mp4"
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    transcode_cmd = [
        ffmpeg,
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
    transcode_result = subprocess.run(transcode_cmd, capture_output=True, text=True, check=False)
    if transcode_result.returncode != 0:
        return None
    return fallback_path


def _frame_extract_command(
    ffmpeg: str,
    source: Path,
    pattern: Path,
    fps: float,
    start: float | None,
    end: float | None,
) -> list[str]:
    command = [ffmpeg, "-y", "-hwaccel", "none", "-analyzeduration", "100M", "-probesize", "100M"]
    if start is not None:
        command.extend(["-ss", str(start)])
    if end is not None and start is not None:
        command.extend(["-t", str(max(0.0, end - start))])
    command.extend(["-i", str(source), "-vf", f"fps={fps}", "-q:v", "3", str(pattern)])
    return command


def _ffmpeg_executables() -> list[str]:
    candidates: list[str] = []
    local = Path("tools/ffmpeg/bin/ffmpeg.exe").resolve()
    if local.exists():
        candidates.append(str(local))
    executable = shutil.which("ffmpeg")
    if executable and executable not in candidates:
        candidates.append(executable)
    try:
        import imageio_ffmpeg

        imageio_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if imageio_exe not in candidates:
            candidates.append(imageio_exe)
    except Exception as exc:  # noqa: BLE001
        if not candidates:
            raise RuntimeError("ffmpeg is required. Install ffmpeg or `pip install imageio-ffmpeg`.") from exc
    if not candidates:
        raise RuntimeError("ffmpeg is required. Install ffmpeg and ensure it is in PATH.")
    return candidates


def frames_from_directory(
    frame_dir: str | Path,
    fps: float = 1.0,
    start: float = 0.0,
    video_id: str = "video",
) -> list[VideoFrame]:
    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
    seconds_per_frame = 1.0 / fps if fps > 0 else 1.0
    frames: list[VideoFrame] = []
    for index, path in enumerate(frame_paths):
        timestamp = start + index * seconds_per_frame
        frames.append(VideoFrame(frame_id=f"f_{math.floor(timestamp * 1000):010d}", timestamp=timestamp, path=str(path)))
    return frames
