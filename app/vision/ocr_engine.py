from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.models import OCRResult, VideoFrame

logger = logging.getLogger(__name__)


class OCREngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run_ocr(self, frames: list[VideoFrame]) -> list[OCRResult]:
        engine = self.settings.ocr_engine.lower()
        if self.settings.ocr_provider == "none" or engine == "none":
            logger.info("OCR disabled; returning empty OCR results.")
            return [self._empty(frame) for frame in frames]
        if engine == "tesseract":
            return [self._run_tesseract(frame) for frame in frames]
        if engine == "easyocr":
            return self._run_easyocr(frames)
        logger.warning("Unsupported OCR engine '%s'; returning empty OCR results.", self.settings.ocr_engine)
        return [self._empty(frame) for frame in frames]

    def _run_tesseract(self, frame: VideoFrame) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(frame.path), lang=self.settings.ocr_lang).strip()
            return OCRResult(frame_id=frame.frame_id, timestamp=frame.timestamp, text=text, confidence=None, boxes=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract OCR failed for %s: %s", frame.frame_id, exc)
            return self._empty(frame)

    def _run_easyocr(self, frames: list[VideoFrame]) -> list[OCRResult]:
        try:
            import easyocr

            langs = [item.strip() for item in self.settings.ocr_lang.split(",") if item.strip()] or ["en"]
            reader = easyocr.Reader(langs, gpu=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR initialization failed: %s", exc)
            return [self._empty(frame) for frame in frames]

        results: list[OCRResult] = []
        for frame in frames:
            try:
                raw = reader.readtext(frame.path)
                text = "\n".join(item[1] for item in raw).strip()
                confidences = [float(item[2]) for item in raw if len(item) >= 3]
                confidence = sum(confidences) / len(confidences) if confidences else None
                boxes = [_box_to_dict(item[0]) for item in raw if item]
                results.append(OCRResult(frame_id=frame.frame_id, timestamp=frame.timestamp, text=text, confidence=confidence, boxes=boxes))
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR failed for %s: %s", frame.frame_id, exc)
                results.append(self._empty(frame))
        return results

    def _empty(self, frame: VideoFrame) -> OCRResult:
        return OCRResult(frame_id=frame.frame_id, timestamp=frame.timestamp, text="", confidence=None, boxes=None)


def _box_to_dict(box: object) -> dict[str, float]:
    try:
        points = list(box)  # type: ignore[arg-type]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}
    except Exception:
        return {}


def save_ocr_results(results: list[OCRResult], video_dir: str | Path) -> Path:
    path = Path(video_dir) / "ocr.json"
    path.write_text(json.dumps([result.model_dump() for result in results], ensure_ascii=False, indent=2), encoding="utf-8")
    return path
