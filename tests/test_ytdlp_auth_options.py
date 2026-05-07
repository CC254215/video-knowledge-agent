from __future__ import annotations

from app.config import get_settings
from app.ingestion.downloader import (
    _DISABLED_BROWSER_COOKIE_SPECS,
    _DISABLED_COOKIE_FILES,
    _apply_ytdlp_auth_opts,
    _call_ytdlp_with_cookie_fallback,
    _is_valid_netscape_cookie_file,
    _parse_cookies_from_browser,
    classify_ytdlp_error,
)


def test_parse_cookies_from_browser_browser_only() -> None:
    assert _parse_cookies_from_browser("chrome") == ("chrome",)


def test_parse_cookies_from_browser_with_profile() -> None:
    assert _parse_cookies_from_browser("chrome:Profile 1") == ("chrome", "Profile 1")


def test_apply_ytdlp_auth_opts_uses_cookies_from_browser(monkeypatch) -> None:
    get_settings.cache_clear()
    _DISABLED_BROWSER_COOKIE_SPECS.clear()
    _DISABLED_COOKIE_FILES.clear()
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "edge")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", "missing-cookies.txt")

    opts = _apply_ytdlp_auth_opts({})

    assert opts["cookiesfrombrowser"] == ("edge",)
    assert "cookiefile" not in opts
    get_settings.cache_clear()


def test_ytdlp_cookie_copy_failure_retries_without_browser_cookies(monkeypatch) -> None:
    get_settings.cache_clear()
    _DISABLED_BROWSER_COOKIE_SPECS.clear()
    _DISABLED_COOKIE_FILES.clear()
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "chrome")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", "")
    calls = []

    def operation(opts):
        calls.append(opts)
        if "cookiesfrombrowser" in opts:
            raise RuntimeError("ERROR: Could not copy Chrome cookie database.")
        return {"ok": True}

    result = _call_ytdlp_with_cookie_fallback(operation, {"cookiesfrombrowser": ("chrome",), "quiet": True})

    assert result == {"ok": True}
    assert calls == [
        {"cookiesfrombrowser": ("chrome",), "quiet": True},
        {"quiet": True},
    ]
    assert "chrome" in _DISABLED_BROWSER_COOKIE_SPECS
    get_settings.cache_clear()


def test_ytdlp_non_cookie_error_is_not_retried() -> None:
    calls = 0

    def operation(opts):
        nonlocal calls
        calls += 1
        raise RuntimeError("network unavailable")

    try:
        _call_ytdlp_with_cookie_fallback(operation, {"cookiesfrombrowser": ("chrome",)})
    except RuntimeError as exc:
        assert "network unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == 1


def test_ytdlp_invalid_cookie_file_is_not_used_after_browser_failure(tmp_path, monkeypatch) -> None:
    get_settings.cache_clear()
    _DISABLED_BROWSER_COOKIE_SPECS.clear()
    _DISABLED_COOKIE_FILES.clear()
    cookie_file = tmp_path / "youtube_cookies.txt"
    cookie_file.write_text("not netscape cookies", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookie_file))
    calls = []

    def operation(opts):
        calls.append(opts)
        if "cookiesfrombrowser" in opts:
            raise RuntimeError("ERROR: Could not copy Chrome cookie database.")
        if "cookiefile" in opts:
            raise RuntimeError(f"ERROR: '{opts['cookiefile']}' does not look like a Netscape format cookies file")
        return {"ok": True}

    result = _call_ytdlp_with_cookie_fallback(operation, {"cookiesfrombrowser": ("chrome",), "quiet": True})

    assert result == {"ok": True}
    assert calls == [
        {"cookiesfrombrowser": ("chrome",), "quiet": True},
        {"quiet": True},
    ]
    assert str(cookie_file) not in _DISABLED_COOKIE_FILES
    get_settings.cache_clear()


def test_invalid_cookie_file_classification() -> None:
    assert classify_ytdlp_error(RuntimeError("does not look like a Netscape format cookies file")) == "invalid_cookie_file"


def test_apply_ytdlp_auth_opts_skips_env_file_mistaken_as_cookie_file(tmp_path, monkeypatch) -> None:
    get_settings.cache_clear()
    _DISABLED_BROWSER_COOKIE_SPECS.clear()
    _DISABLED_COOKIE_FILES.clear()
    cookie_file = tmp_path / "youtube_cookies.txt"
    cookie_file.write_text(f"YTDLP_COOKIES_FILE={cookie_file}", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", str(cookie_file))

    opts = _apply_ytdlp_auth_opts({})

    assert "cookiefile" not in opts
    assert str(cookie_file) in _DISABLED_COOKIE_FILES
    assert not _is_valid_netscape_cookie_file(cookie_file)
    get_settings.cache_clear()


def test_valid_netscape_cookie_file_is_accepted(tmp_path) -> None:
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tSID\tvalue\n",
        encoding="utf-8",
    )

    assert _is_valid_netscape_cookie_file(cookie_file)
