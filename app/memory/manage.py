"""Memory administration: python -m app.memory.manage --help."""
from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.memory.knowledge_catalog import KnowledgeCatalog, historical_context
from app.memory.mempalace_adapter import create_memory_adapter
from app.models import Storyline, VideoMetadata
from app.transcript.multimodal_segmenter import load_multimodal_segments


def backfill(catalog: KnowledgeCatalog, video_id: str | None = None) -> dict:
    output = {"videos": 0, "items": 0, "errors": []}
    for directory in sorted(catalog.settings.videos_dir.glob("*")):
        if not directory.is_dir() or (video_id and directory.name != video_id):
            continue
        if not (directory / "metadata.json").exists() or not (directory / "multimodal_segments.json").exists():
            continue
        try:
            metadata = VideoMetadata.model_validate_json((directory / "metadata.json").read_text(encoding="utf-8"))
            storyline_path = directory / "storyline.json"
            storyline = Storyline.model_validate_json(storyline_path.read_text(encoding="utf-8")) if storyline_path.exists() else Storyline(video_id=metadata.video_id, query="", nodes=[])
            segments = load_multimodal_segments(directory / "multimodal_segments.json")
            output["items"] += catalog.ingest_video(metadata, storyline, segments)
            output["videos"] += 1
        except Exception as exc:
            output["errors"].append({"video_id": directory.name, "error": str(exc)})
    return output


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("list")
    search = sub.add_parser("search")
    search.add_argument("question")
    search.add_argument("--video-id", required=True)
    smoke = sub.add_parser("smoke-qa")
    smoke.add_argument("question")
    smoke.add_argument("--video-id", required=True)
    fill = sub.add_parser("backfill")
    fill.add_argument("--video-id")
    sync = sub.add_parser("sync")
    sync.add_argument("--limit", type=int, default=100)
    topics = sub.add_parser("topics")
    topics.add_argument("video_id")
    topics.add_argument("--set", nargs="+", dest="labels")
    pin = sub.add_parser("pin")
    pin.add_argument("memory_id")
    pin.add_argument("--off", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    catalog = KnowledgeCatalog(settings)
    adapter = None
    try:
        if args.command == "smoke-qa":
            from app.reasoning.conversation_agent import ConversationAgent
            from app.retrieval.chroma_memory import ChromaMemoryStore

            directory = (settings.videos_dir / args.video_id).resolve()
            if not directory.is_relative_to(settings.videos_dir.resolve()):
                raise ValueError("Invalid video id")
            agent = ConversationAgent(args.video_id, directory, settings, ChromaMemoryStore(settings, use_persistent=False))
            turn = agent.answer(args.question, persist=False)
            result = turn.model_dump(mode="json")
            (settings.data_dir / "memory" / "smoke_qa.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        elif args.command == "search":
            path = settings.videos_dir / args.video_id / "metadata.json"
            title = VideoMetadata.model_validate_json(path.read_text(encoding="utf-8")).title if path.is_file() else args.video_id
            result = historical_context(args.question, args.video_id, title, settings, current_chars=14000)
        elif args.command == "list":
            with catalog.connect() as db:
                result = {"videos": [{"video_id": row[0], "topics": json.loads(row[1])} for row in db.execute("SELECT video_id,topics FROM video_topics WHERE video_id IN (SELECT video_id FROM knowledge WHERE active=1) ORDER BY video_id")]}
        elif args.command == "backfill":
            result = backfill(catalog, args.video_id)
        elif args.command == "topics":
            if args.labels:
                catalog.assign_topics(args.video_id, args.labels, manual=True)
                backfill(catalog, args.video_id)
            result = {"topics": catalog.topics(args.video_id)}
        elif args.command == "pin":
            catalog.pin(args.memory_id, not args.off)
            result = {"saved": True}
        else:
            if settings.mempalace_provider == "noop":
                result = {"status": "disabled", **catalog.stats()}
            else:
                adapter = create_memory_adapter(settings)
                result = catalog.sync(adapter, args.limit) if args.command == "sync" else {"backend": adapter.status() if hasattr(adapter, "status") else "http", **catalog.stats()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("error") or result.get("errors") or (isinstance(result.get("backend"), dict) and result["backend"].get("error")) else 0
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
