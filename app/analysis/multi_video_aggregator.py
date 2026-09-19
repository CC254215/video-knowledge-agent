from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.memory.mempalace_adapter import create_memory_adapter
from app.memory.obsidian_writer import safe_filename
from app.models import MultimodalSegment, Storyline, StorylineNode, VideoMetadata
from app.retrieval.embeddings import embed_texts
from app.transcript.multimodal_segmenter import load_multimodal_segments


@dataclass
class ViewpointRecord:
    record_id: str
    video_id: str
    video_title: str
    author: str
    created_at: str
    node_id: str
    text: str
    evidence_segment_ids: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    cluster_id: int | None = None


@dataclass
class TopicCluster:
    cluster_id: int
    description: str
    records: list[ViewpointRecord]
    consensus: list[str]
    disagreements: list[str]
    similarity_table: list[dict[str, Any]]
    author_changes: list[dict[str, Any]]


class MultiVideoAggregator:
    """Aggregate processed video storylines into topic notes.

    Expected input layout:
    data/videos/{video_id}/metadata.json
    data/videos/{video_id}/storyline.json
    data/videos/{video_id}/multimodal_segments.json
    """

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str = "multi_video_viewpoints",
        consensus_threshold: float = 0.78,
        disagreement_threshold: float = 0.45,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.consensus_threshold = consensus_threshold
        self.disagreement_threshold = disagreement_threshold

    def run(
        self,
        video_ids: list[str] | None = None,
        k: int | None = None,
        export_obsidian: bool = True,
    ) -> dict[str, Any]:
        records = self.load_viewpoints(video_ids)
        if not records:
            return {"topics": [], "message": "No storyline viewpoints found."}

        self.embed_records(records)
        chroma_stats = self.write_chroma(records)
        labels = self.cluster_records(records, k=k)
        for record, label in zip(records, labels, strict=False):
            record.cluster_id = int(label)

        topics = self.build_topics(records)
        exported = self.export_topics_to_obsidian(topics) if export_obsidian and self.settings.obsidian_vault_path else []
        memory_saved = self.save_to_memory(topics)
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "topic_count": len(topics),
            "chroma": chroma_stats,
            "topics": [self.topic_to_dict(topic) for topic in topics],
            "obsidian_paths": [str(path) for path in exported],
            "memory_saved": memory_saved,
        }

    def load_viewpoints(self, video_ids: list[str] | None = None) -> list[ViewpointRecord]:
        videos_root = self.settings.videos_dir
        selected = set(video_ids or [])
        records: list[ViewpointRecord] = []
        for video_dir in sorted(videos_root.iterdir() if videos_root.exists() else []):
            if not video_dir.is_dir():
                continue
            if selected and video_dir.name not in selected:
                continue

            metadata_path = video_dir / "metadata.json"
            storyline_path = video_dir / "storyline.json"
            if not metadata_path.exists() or not storyline_path.exists():
                continue

            metadata = VideoMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
            storyline = Storyline.model_validate(json.loads(storyline_path.read_text(encoding="utf-8")))
            segments = self._load_segments_by_id(video_dir)
            for node in storyline.nodes:
                text = self._viewpoint_text(node)
                if not text.strip():
                    continue
                evidence_ids = node.evidence_segment_ids or node.speech_evidence_ids
                if not evidence_ids and node.status.value == "unsupported":
                    continue
                enriched_text = self._append_evidence_text(text, evidence_ids, segments)
                records.append(
                    ViewpointRecord(
                        record_id=f"{metadata.video_id}:{node.node_id}",
                        video_id=metadata.video_id,
                        video_title=metadata.title,
                        author=metadata.author or "Unknown",
                        created_at=metadata.created_at.isoformat(),
                        node_id=node.node_id,
                        text=enriched_text,
                        evidence_segment_ids=evidence_ids,
                    )
                )
        return records

    def embed_records(self, records: list[ViewpointRecord]) -> None:
        vectors = embed_texts([record.text for record in records], self.settings)
        for record, vector in zip(records, vectors, strict=False):
            record.embedding = vector

    def write_chroma(self, records: list[ViewpointRecord]) -> dict[str, Any]:
        """Persist all viewpoint embeddings into one cross-video Chroma collection."""
        try:
            import chromadb  # type: ignore

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            collection = client.get_or_create_collection(self.collection_name, metadata={"scope": "multi_video"})
            collection.upsert(
                ids=[record.record_id for record in records],
                documents=[record.text for record in records],
                embeddings=[record.embedding for record in records],
                metadatas=[
                    {
                        "video_id": record.video_id,
                        "video_title": record.video_title,
                        "author": record.author,
                        "node_id": record.node_id,
                        "created_at": record.created_at,
                    }
                    for record in records
                ],
            )
            return {"collection": self.collection_name, "count": collection.count(), "using_chroma": True}
        except Exception as exc:  # noqa: BLE001
            return {"collection": self.collection_name, "count": len(records), "using_chroma": False, "warning": str(exc)}

    def cluster_records(self, records: list[ViewpointRecord], k: int | None = None) -> list[int]:
        matrix = np.array([record.embedding for record in records], dtype=float)
        if len(records) == 1:
            return [0]

        cluster_count = k or min(max(2, round(math.sqrt(len(records)))), len(records))
        try:
            from sklearn.cluster import KMeans

            labels = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(matrix)
            return [int(label) for label in labels]
        except Exception:
            return self._fallback_cosine_clusters(matrix, threshold=0.72)

    def build_topics(self, records: list[ViewpointRecord]) -> list[TopicCluster]:
        topics: list[TopicCluster] = []
        for cluster_id in sorted({record.cluster_id for record in records if record.cluster_id is not None}):
            cluster_records = [record for record in records if record.cluster_id == cluster_id]
            table = self.similarity_table(cluster_records)
            consensus, disagreements = self.consensus_and_disagreements(cluster_records, table)
            topics.append(
                TopicCluster(
                    cluster_id=int(cluster_id),
                    description=self.describe_cluster(cluster_records),
                    records=cluster_records,
                    consensus=consensus,
                    disagreements=disagreements,
                    similarity_table=table,
                    author_changes=self.author_change_report(cluster_records),
                )
            )
        return topics

    def similarity_table(self, records: list[ViewpointRecord]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                rows.append(
                    {
                        "left": left.record_id,
                        "right": right.record_id,
                        "left_video": left.video_id,
                        "right_video": right.video_id,
                        "left_author": left.author,
                        "right_author": right.author,
                        "similarity": round(cosine(left.embedding, right.embedding), 4),
                    }
                )
        return rows

    def consensus_and_disagreements(
        self,
        records: list[ViewpointRecord],
        table: list[dict[str, Any]],
    ) -> tuple[list[str], list[str]]:
        consensus: list[str] = []
        disagreements: list[str] = []
        for row in table:
            left = self._record_by_id(records, row["left"])
            right = self._record_by_id(records, row["right"])
            if not left or not right:
                continue
            if row["similarity"] >= self.consensus_threshold:
                consensus.append(f"{left.video_title} / {right.video_title}: {short(left.text)} -> {short(right.text)}")
            elif row["similarity"] <= self.disagreement_threshold:
                disagreements.append(f"{left.video_title} / {right.video_title}: {short(left.text)} -> {short(right.text)}")
        return consensus[:12], disagreements[:12]

    def author_change_report(self, records: list[ViewpointRecord]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        by_author: dict[str, list[ViewpointRecord]] = {}
        for record in records:
            by_author.setdefault(record.author, []).append(record)

        for author, author_records in by_author.items():
            ordered = sorted(author_records, key=lambda item: item.created_at)
            if len(ordered) < 2:
                continue
            transitions = []
            for prev, curr in zip(ordered, ordered[1:], strict=False):
                sim = cosine(prev.embedding, curr.embedding)
                transitions.append(
                    {
                        "from_video": prev.video_id,
                        "to_video": curr.video_id,
                        "from_node": prev.node_id,
                        "to_node": curr.node_id,
                        "similarity": round(sim, 4),
                        "change": "stable" if sim >= self.consensus_threshold else "changed",
                    }
                )
            output.append({"author": author, "transitions": transitions})
        return output

    def export_topics_to_obsidian(self, topics: list[TopicCluster]) -> list[Path]:
        vault = self.settings.obsidian_vault_path
        if not vault:
            return []
        topic_dir = Path(vault) / "40_MOCs" / "Video_Topics"
        topic_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for topic in topics:
            filename = safe_filename(f"Topic {topic.cluster_id} - {topic.description[:60]}") + ".md"
            path = unique_path(topic_dir / filename)
            path.write_text(render_topic_note(topic), encoding="utf-8")
            paths.append(path)
        return paths

    def save_to_memory(self, topics: list[TopicCluster]) -> int:
        """Save model-derived topic summaries through the memory adapter boundary."""
        if self.settings.mempalace_provider == "noop":
            return 0
        adapter = None
        saved = 0
        try:
            adapter = create_memory_adapter(self.settings)
            for topic in topics:
                adapter.save_model_summary(
                    {
                        "summary_type": "multi_video_topic_cluster",
                        "cluster_id": topic.cluster_id,
                        "description": topic.description,
                        "video_ids": sorted({record.video_id for record in topic.records}),
                        "node_ids": [record.node_id for record in topic.records],
                        "consensus": topic.consensus,
                        "disagreements": topic.disagreements,
                        "status": "similarity_heuristic_only",
                        "excluded_from_factual_retrieval": True,
                    }
                )
                saved += 1
        except Exception:
            # Topic notes remain useful when the optional memory backend is offline.
            pass
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
        return saved

    def describe_cluster(self, records: list[ViewpointRecord]) -> str:
        words: list[str] = []
        for record in records:
            normalized = record.text.replace("，", " ").replace("。", " ").replace(",", " ").replace(".", " ")
            words.extend(token for token in normalized.split() if len(token) >= 2)
        if not words:
            return "Untitled Topic"
        counts = {word: words.count(word) for word in set(words)}
        return " / ".join(word for word, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5])

    def topic_to_dict(self, topic: TopicCluster) -> dict[str, Any]:
        return {
            "cluster_id": topic.cluster_id,
            "description": topic.description,
            "videos": sorted({record.video_id for record in topic.records}),
            "nodes": [{"video_id": record.video_id, "node_id": record.node_id, "author": record.author} for record in topic.records],
            "consensus": topic.consensus,
            "disagreements": topic.disagreements,
            "similarity_table": topic.similarity_table,
            "author_changes": topic.author_changes,
        }

    def _load_segments_by_id(self, video_dir: Path) -> dict[str, MultimodalSegment]:
        path = video_dir / "multimodal_segments.json"
        if not path.exists():
            return {}
        return {segment.segment_id: segment for segment in load_multimodal_segments(path)}

    def _append_evidence_text(self, text: str, evidence_ids: list[str], segments: dict[str, MultimodalSegment]) -> str:
        snippets = []
        for segment_id in evidence_ids[:3]:
            segment = segments.get(segment_id)
            if segment and segment.transcript_text:
                snippets.append(short(segment.transcript_text, 220))
        return text + ("\nEvidence: " + " | ".join(snippets) if snippets else "")

    def _viewpoint_text(self, node: StorylineNode) -> str:
        parts = [node.title or node.topic, node.claim, node.summary or "", " ".join(node.key_points)]
        return "\n".join(part for part in parts if part and part.strip())

    def _record_by_id(self, records: list[ViewpointRecord], record_id: str) -> ViewpointRecord | None:
        return next((record for record in records if record.record_id == record_id), None)

    def _fallback_cosine_clusters(self, matrix: np.ndarray, threshold: float) -> list[int]:
        labels = [-1] * len(matrix)
        next_label = 0
        for index in range(len(matrix)):
            if labels[index] != -1:
                continue
            labels[index] = next_label
            for other in range(index + 1, len(matrix)):
                if labels[other] == -1 and cosine(matrix[index].tolist(), matrix[other].tolist()) >= threshold:
                    labels[other] = next_label
            next_label += 1
        return labels


def cosine(left: list[float], right: list[float]) -> float:
    left_arr = np.array(left, dtype=float)
    right_arr = np.array(right, dtype=float)
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    return float(np.dot(left_arr, right_arr) / denom) if denom else 0.0


def short(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    return path.with_name(f"{path.stem} - {stamp}{path.suffix}")


def render_topic_note(topic: TopicCluster) -> str:
    videos = sorted({record.video_id for record in topic.records})
    lines = [
        "---",
        "type: video-topic-cluster",
        f"cluster_id: {topic.cluster_id}",
        f"generated_at: {datetime.utcnow().isoformat()}",
        f"video_count: {len(videos)}",
        f"node_count: {len(topic.records)}",
        "---",
        "",
        f"# Topic {topic.cluster_id}: {topic.description}",
        "",
        "## 视频与节点",
        "",
    ]
    for record in topic.records:
        lines.append(f"- `{record.video_id}` `{record.node_id}` {record.author}: {record.video_title}")

    lines.extend(["", "## 共识观点", ""])
    if topic.consensus:
        lines.extend(f"- {item}" for item in topic.consensus)
    else:
        lines.append("- 暂未发现高相似共识。")

    lines.extend(["", "## 分歧观点", ""])
    if topic.disagreements:
        lines.extend(f"- {item}" for item in topic.disagreements)
    else:
        lines.append("- 暂未发现明显分歧。")

    lines.extend(["", "## 相似度表", ""])
    for row in topic.similarity_table[:30]:
        lines.append(f"- `{row['left']}` -> `{row['right']}` similarity={row['similarity']}")

    lines.extend(["", "## 作者观点变化", ""])
    if not topic.author_changes:
        lines.append("- 暂无同一作者跨视频变化记录。")
    for item in topic.author_changes:
        lines.append(f"- {item['author']}")
        for transition in item["transitions"]:
            lines.append(
                f"  - `{transition['from_video']}` -> `{transition['to_video']}` "
                f"similarity={transition['similarity']} status={transition['change']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate processed videos into cross-video topic notes.")
    parser.add_argument("--video-id", action="append", dest="video_ids", help="Video id to include. Repeatable. Defaults to all processed videos.")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters. Defaults to sqrt(N).")
    parser.add_argument("--no-obsidian", action="store_true", help="Do not write Obsidian notes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = MultiVideoAggregator().run(video_ids=args.video_ids, k=args.k, export_obsidian=not args.no_obsidian)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
