from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models import VideoMetadata
from app.runtime.scheduler import get_scheduler

logger = logging.getLogger(__name__)
_DISABLED_BROWSER_COOKIE_SPECS: set[str] = set()
_DISABLED_COOKIE_FILES: set[str] = set()


class DownloadError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _apply_ytdlp_auth_opts(ydl_opts: dict[str, Any]) -> dict[str, Any]:
    """Inject authentication options for sites requiring login state."""
    settings = get_settings()
    cookies_from_browser = settings.ytdlp_cookies_from_browser
    if cookies_from_browser and cookies_from_browser not in _DISABLED_BROWSER_COOKIE_SPECS:
        ydl_opts["cookiesfrombrowser"] = _parse_cookies_from_browser(cookies_from_browser)
        return ydl_opts
    if cookies_from_browser in _DISABLED_BROWSER_COOKIE_SPECS:
        logger.info("Skipping disabled YTDLP_COOKIES_FROM_BROWSER=%s for this process.", cookies_from_browser)
    cookies_file = settings.ytdlp_cookies_file
    if cookies_file:
        cookie_path = Path(cookies_file)
        if str(cookie_path) in _DISABLED_COOKIE_FILES:
            logger.info("Skipping disabled YTDLP_COOKIES_FILE=%s for this process.", cookie_path)
        elif _is_valid_netscape_cookie_file(cookie_path):
            ydl_opts["cookiefile"] = str(cookie_path)
        elif cookie_path.exists() and cookie_path.is_file():
            logger.warning(
                "YTDLP_COOKIES_FILE is not a valid Netscape cookies file and will be ignored: %s",
                cookie_path,
            )
            _DISABLED_COOKIE_FILES.add(str(cookie_path))
        else:
            logger.warning("YTDLP_COOKIES_FILE is set but file does not exist: %s", cookie_path)
    return ydl_opts


def _retry_opts_without_browser_cookies(ydl_opts: dict[str, Any]) -> dict[str, Any]:
    opts = dict(ydl_opts)
    if "cookiesfrombrowser" not in opts:
        return opts
    opts.pop("cookiesfrombrowser", None)
    settings = get_settings()
    if settings.ytdlp_cookies_from_browser:
        _DISABLED_BROWSER_COOKIE_SPECS.add(settings.ytdlp_cookies_from_browser)
    cookies_file = settings.ytdlp_cookies_file
    if cookies_file:
        cookie_path = Path(cookies_file)
        if _is_valid_netscape_cookie_file(cookie_path):
            opts["cookiefile"] = str(cookie_path)
    return opts


def _call_ytdlp_with_cookie_fallback(operation, ydl_opts: dict[str, Any]):
    try:
        return operation(ydl_opts)
    except Exception as exc:  # noqa: BLE001
        if "cookiesfrombrowser" not in ydl_opts or not _is_browser_cookie_copy_error(exc):
            raise
        fallback_opts = _retry_opts_without_browser_cookies(ydl_opts)
        logger.warning(
            "yt-dlp could not read browser cookies; retrying without cookiesfrombrowser. "
            "Close the browser or set YTDLP_COOKIES_FILE for authenticated videos. Original error: %s",
            exc,
        )
        try:
            return operation(fallback_opts)
        except Exception as fallback_exc:  # noqa: BLE001
            if "cookiefile" not in fallback_opts or not _is_invalid_cookie_file_error(fallback_exc):
                raise
            retry_opts = dict(fallback_opts)
            bad_cookie_file = retry_opts.pop("cookiefile", None)
            if bad_cookie_file:
                _DISABLED_COOKIE_FILES.add(str(bad_cookie_file))
            logger.warning(
                "yt-dlp cookie file is invalid and will be ignored for this attempt: %s. "
                "Export cookies in Netscape format if the video requires login. Original error: %s",
                bad_cookie_file,
                fallback_exc,
            )
            return operation(retry_opts)


def _parse_cookies_from_browser(value: str) -> tuple[str, ...]:
    """Parse a compact cookies-from-browser setting for yt-dlp's Python API.

    Supported values:
    - chrome
    - edge
    - firefox
    - chrome:Profile 1
    """
    parts = [part.strip() for part in value.split(":", 1)]
    browser = parts[0].lower()
    if browser not in {"brave", "chrome", "chromium", "edge", "firefox", "opera", "safari", "vivaldi", "whale"}:
        logger.warning("Unsupported YTDLP_COOKIES_FROM_BROWSER browser=%s; passing through to yt-dlp", browser)
    if len(parts) == 1 or not parts[1]:
        return (browser,)
    return (browser, parts[1])


def classify_ytdlp_error(exc: Exception) -> str:
    text = str(exc).lower()
    if _is_invalid_cookie_file_error(exc):
        return "invalid_cookie_file"
    if "cookie" in text or "sign in" in text or "login" in text:
        return "cookie_required"
    if "unsupported url" in text or "no suitable extractor" in text:
        return "unsupported_site"
    if "invalid url" in text or "not a valid url" in text:
        return "invalid_url"
    return "download_failed"


def _is_browser_cookie_copy_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "could not copy" in text and "cookie" in text and "database" in text


def _is_invalid_cookie_file_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "does not look like" in text and "netscape format cookies file" in text


def _is_valid_netscape_cookie_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    meaningful = [line.strip() for line in lines if line.strip()]
    if not meaningful:
        return False
    if any(line.startswith("# Netscape HTTP Cookie File") for line in meaningful[:5]):
        return True
    for line in meaningful:
        if line.startswith("#"):
            continue
        if line.startswith(("YTDLP_", "http://", "https://")) or "=" in line and "\t" not in line:
            return False
        if len(line.split("\t")) >= 7:
            return True
    return False


def fetch_url_metadata(url: str, output_dir: Path) -> tuple[VideoMetadata, dict[str, Any]]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("download_failed", "yt-dlp is not installed. Install project dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = _apply_ytdlp_auth_opts({
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["zh-Hans", "zh", "en"],
        "quiet": True,
        "no_warnings": True,
    })
    try:
        info = get_scheduler().call(
            "ytdlp",
            lambda: _call_ytdlp_with_cookie_fallback(
                lambda opts: _extract_info(yt_dlp, opts, url, download=False),
                ydl_opts,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        code = classify_ytdlp_error(exc)
        logger.warning("yt-dlp metadata fetch failed: %s", exc)
        raise DownloadError(code, f"yt-dlp failed to fetch URL metadata: {exc}") from exc

    video_id = str(info.get("id") or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12])
    metadata = VideoMetadata(
        video_id=video_id,
        title=info.get("title") or "Untitled Video",
        author=info.get("uploader") or info.get("channel"),
        source=info.get("extractor_key") or info.get("extractor") or "url",
        url=info.get("webpage_url") or url,
        duration=float(info["duration"]) if info.get("duration") else None,
        language=info.get("language"),
    )
    return metadata, info


def download_subtitle_file(url: str, output_dir: Path) -> Path | None:
    """Download the best available human subtitle, then automatic subtitle.

    The returned path is one of vtt/srt/json3 when available. Empty result means subtitles are not
    available; callers should fall back to ASR.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("download_failed", "yt-dlp is not installed. Install project dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    common = _apply_ytdlp_auth_opts({
        "skip_download": True,
        "subtitleslangs": ["zh-Hans", "zh-CN", "zh", "en", "all"],
        "subtitlesformat": "vtt/srt/json3",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    })
    for automatic in (False, True):
        before = set(output_dir.glob("*"))
        opts = dict(common)
        opts.update({"writesubtitles": not automatic, "writeautomaticsub": automatic})
        try:
            get_scheduler().call(
                "ytdlp",
                lambda opts=opts: _call_ytdlp_with_cookie_fallback(
                    lambda retry_opts: _extract_info(yt_dlp, retry_opts, url, download=True),
                    opts,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("yt-dlp subtitle attempt failed automatic=%s: %s", automatic, exc)
            continue
        new_files = [path for path in output_dir.glob("*") if path not in before and path.suffix.lower() in {".vtt", ".srt", ".json3"}]
        if not new_files:
            new_files = sorted(output_dir.glob("*.vtt")) + sorted(output_dir.glob("*.srt")) + sorted(output_dir.glob("*.json3"))
        if new_files:
            return sorted(new_files, key=lambda path: (0 if re.search(r"zh|cn|hans", path.name, re.I) else 1, path.suffix))[0]
    return None


def download_audio_for_asr(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("download_failed", "yt-dlp is not installed. Install project dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _first_existing(output_dir, ("source_audio.*",))
    if existing:
        return existing
    ydl_opts = _apply_ytdlp_auth_opts({
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "source_audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    })
    ffmpeg_location = _local_ffmpeg_location()
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location
    try:
        info, filename = get_scheduler().call(
            "ytdlp",
            lambda: _call_ytdlp_with_cookie_fallback(
                lambda opts: _download_and_filename(yt_dlp, opts, url),
                ydl_opts,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        code = "audio_download_failed" if classify_ytdlp_error(exc) == "download_failed" else classify_ytdlp_error(exc)
        raise DownloadError(code, f"yt-dlp failed to download audio: {exc}") from exc
    return filename


def download_video_file(url: str, output_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise DownloadError("download_failed", "yt-dlp is not installed. Install project dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = _first_existing(output_dir, ("source_video.mp4", "source_video.*"))
    if existing:
        return existing
    ydl_opts = _apply_ytdlp_auth_opts({
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "source_video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    })
    ffmpeg_location = _local_ffmpeg_location()
    if ffmpeg_location:
        ydl_opts["ffmpeg_location"] = ffmpeg_location
    try:
        info, filename = get_scheduler().call(
            "ytdlp",
            lambda: _call_ytdlp_with_cookie_fallback(
                lambda opts: _download_and_filename(yt_dlp, opts, url),
                ydl_opts,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        code = classify_ytdlp_error(exc)
        raise DownloadError(code, f"yt-dlp failed to download video: {exc}") from exc
    mp4 = output_dir / "source_video.mp4"
    return mp4 if mp4.exists() else filename


def _local_ffmpeg_location() -> str | None:
    local_bin = Path("tools/ffmpeg/bin").resolve()
    if (local_bin / "ffmpeg.exe").exists():
        return str(local_bin)
    return None


def _first_existing(output_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path for path in output_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _extract_info(yt_dlp_module: Any, opts: dict[str, Any], url: str, download: bool) -> dict[str, Any]:
    with yt_dlp_module.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=download)


def _download_and_filename(yt_dlp_module: Any, opts: dict[str, Any], url: str) -> tuple[dict[str, Any], Path]:
    with yt_dlp_module.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return info, Path(ydl.prepare_filename(info))
