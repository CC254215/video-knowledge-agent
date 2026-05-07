from __future__ import annotations

import hashlib
from pathlib import Path

from app.models import VideoMetadata


def metadata_from_local_file(file_path: str | Path) -> VideoMetadata:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Local media file does not exist: {path}")
    digest = _file_sha1(path)[:12]
    return VideoMetadata(
        video_id=f"local_{digest}",
        title=path.stem,
        source="local_file",
        local_path=str(path),
    )


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
