from __future__ import annotations

import hashlib
from pathlib import Path

from app.models import VideoFrame


def compute_perceptual_hash(path: str | Path) -> str:
    """Cheap first-pass frame fingerprint.

    If Pillow is available, use a tiny grayscale average hash. Otherwise fall back to hashing bytes,
    which still removes exact duplicate extracted frames in tests and static-frame cases.
    """
    try:
        from PIL import Image

        image = Image.open(path).convert("L").resize((8, 8))
        pixels = list(image.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join("1" if pixel >= avg else "0" for pixel in pixels)
        return f"{int(bits, 2):016x}"
    except Exception:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return 999
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return 0 if left == right else 999


def dedupe_frames(frames: list[VideoFrame], max_hash_distance: int = 3, min_time_gap: float = 1.5) -> list[VideoFrame]:
    selected: list[VideoFrame] = []
    for frame in frames:
        frame.perceptual_hash = frame.perceptual_hash or compute_perceptual_hash(frame.path)
        duplicate = False
        for kept in selected:
            if kept.perceptual_hash and hamming_distance(frame.perceptual_hash, kept.perceptual_hash) <= max_hash_distance:
                if abs(frame.timestamp - kept.timestamp) <= min_time_gap:
                    duplicate = True
                    break
        if not duplicate:
            selected.append(frame)
    return selected
