from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeMediaURLError(ValueError):
    pass


TRUSTED_MEDIA_HOSTS = {"youtube.com", "youtu.be"}


def validate_public_media_url(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise UnsafeMediaURLError("unsafe_media_url: only http and https URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise UnsafeMediaURLError("unsafe_media_url: URL must contain a public host and no embedded credentials")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeMediaURLError("unsafe_media_url: localhost is not allowed")
    if any(host == trusted or host.endswith(f".{trusted}") for trusted in TRUSTED_MEDIA_HOSTS):
        return value
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise UnsafeMediaURLError(f"unsafe_media_url: host cannot be resolved: {host}") from exc
    if not addresses:
        raise UnsafeMediaURLError(f"unsafe_media_url: host cannot be resolved: {host}")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise UnsafeMediaURLError(f"unsafe_media_url: non-public address is not allowed: {address}")
    return value
