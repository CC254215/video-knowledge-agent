from pathlib import Path

import pytest

from app.config import Settings
from app.ingestion.local_file import metadata_from_local_file
from app.services.video_service import VideoService


def test_upload_supports_mp4_and_mp3(tmp_path: Path):
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"))
    service = VideoService(settings)
    mp4 = tmp_path / "sample.mp4"
    mp3 = tmp_path / "sample.mp3"
    mp4.write_bytes(b"video")
    mp3.write_bytes(b"audio")

    assert service.copy_upload_to_data(str(mp4)).suffix == ".mp4"
    assert service.copy_upload_to_data(str(mp3)).suffix == ".mp3"


def test_upload_rejects_unsupported_format(tmp_path: Path):
    settings = Settings(_env_file=None, DATA_DIR=str(tmp_path / "data"))
    service = VideoService(settings)
    bad = tmp_path / "sample.txt"
    bad.write_text("not media", encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的文件格式"):
        service.copy_upload_to_data(str(bad))


def test_local_file_video_id_is_content_based(tmp_path: Path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same video bytes")
    second.write_bytes(b"same video bytes")

    assert metadata_from_local_file(first).video_id == metadata_from_local_file(second).video_id
