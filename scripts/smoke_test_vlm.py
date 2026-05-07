from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.config import get_settings
from app.models import VideoFrame
from app.vision.vlm_captioner import VLMCaptioner


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test VLM frame caption with a local image.")
    parser.add_argument("image", nargs="?", help="Local image path. If omitted, creates a tiny test image.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    settings.log_runtime_config()
    image_path = Path(args.image) if args.image else _create_test_image()
    frame = VideoFrame(frame_id="smoke_frame", timestamp=0.0, path=str(image_path), selected=True)
    caption = VLMCaptioner(settings).caption_frames([frame])[0]
    print(caption.model_dump_json(indent=2))


def _create_test_image() -> Path:
    from PIL import Image, ImageDraw

    path = Path("data") / "smoke_vlm_test.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 180), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 290, 140), outline=(20, 80, 180), width=4)
    draw.text((55, 75), "Video Knowledge Agent", fill=(0, 0, 0))
    image.save(path, quality=90)
    return path


if __name__ == "__main__":
    main()
