from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.models import ModalityProfile, Storyline, SummaryReport, VideoMetadata


@dataclass
class AppState:
    current_video_id: str | None = None
    current_metadata: VideoMetadata | None = None
    current_summary: SummaryReport | None = None
    current_storyline: Storyline | None = None
    current_modality_profile: ModalityProfile | None = None
    current_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    processing_status: str = "等待输入"
    last_error: str | None = None
    chat_history: list[dict[str, Any]] = field(default_factory=list)

    def load_video(
        self,
        video_id: str,
        metadata: VideoMetadata | None = None,
        summary: SummaryReport | None = None,
        storyline: Storyline | None = None,
        modality_profile: ModalityProfile | None = None,
    ) -> None:
        self.current_video_id = video_id
        self.current_metadata = metadata
        self.current_summary = summary
        self.current_storyline = storyline
        self.current_modality_profile = modality_profile
        self.current_conversation_id = str(uuid.uuid4())
        self.processing_status = "处理完成"
        self.last_error = None
        self.chat_history = []

    def clear_video(self) -> None:
        self.current_video_id = None
        self.current_metadata = None
        self.current_summary = None
        self.current_storyline = None
        self.current_modality_profile = None
        self.current_conversation_id = str(uuid.uuid4())
        self.processing_status = "等待输入"
        self.last_error = None
        self.chat_history = []

    def clear_chat(self) -> None:
        self.chat_history = []
        self.current_conversation_id = str(uuid.uuid4())

    def fail(self, error: str) -> None:
        self.processing_status = "处理失败"
        self.last_error = error
