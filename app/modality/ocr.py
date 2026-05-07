from pathlib import Path


def extract_text_from_frames(frame_paths: list[str | Path]) -> dict[str, str]:
    """Placeholder OCR hook. First MVP does not require visual understanding."""
    return {str(path): "" for path in frame_paths}
