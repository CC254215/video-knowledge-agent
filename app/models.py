from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceStatus(str, Enum):
    supported = "supported"
    weakly_supported = "weakly_supported"
    unsupported = "unsupported"
    needs_review = "needs_review"


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ModalityMode(str, Enum):
    text_dominant = "text_dominant"
    vision_supportive = "vision_supportive"
    vision_critical = "vision_critical"


EvidenceType = Literal["speech", "frame", "frame_caption"]


class VideoMetadata(BaseModel):
    video_id: str
    title: str
    author: str | None = None
    source: str = "unknown"
    url: str | None = None
    local_path: str | None = None
    duration: float | None = None
    language: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CanonicalEntity(BaseModel):
    """A video-level canonical name with auditable ASR aliases."""

    canonical: str
    aliases: list[str] = Field(default_factory=list)
    entity_type: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_sources: list[str] = Field(default_factory=list)
    reason: str | None = None
    needs_review: bool = False


class TranscriptSegment(BaseModel):
    segment_id: str
    start: float
    end: float
    speaker: str | None = None
    text: str
    asr_confidence: float | None = None
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    @field_validator("end")
    @classmethod
    def end_must_not_precede_start(cls, value: float, info: Any) -> float:
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("segment end must be greater than or equal to start")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("segment text cannot be empty")
        return value


class VideoTranscript(BaseModel):
    video_id: str
    segments: list[TranscriptSegment]
    source: Literal["subtitle", "asr", "mock", "unknown"] = "unknown"
    language: str | None = None


class StorylineNode(BaseModel):
    node_id: str
    time_start: float
    time_end: float
    topic: str
    claim: str
    title: str | None = None
    summary: str | None = None
    key_points: list[str] = Field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_segment_ids: list[str] = Field(default_factory=list)
    speech_evidence_ids: list[str] = Field(default_factory=list)
    frame_evidence_ids: list[str] = Field(default_factory=list)
    frame_caption_evidence: list[dict[str, Any]] = Field(default_factory=list)
    ocr_evidence: list[str] = Field(default_factory=list)
    support_modalities: list[EvidenceType] = Field(default_factory=list)
    modality_support: dict[str, float] = Field(default_factory=dict)
    uncertainty: float = Field(ge=0.0, le=1.0)
    status: EvidenceStatus

    @model_validator(mode="after")
    def enforce_status_for_missing_evidence(self) -> "StorylineNode":
        has_evidence = bool(self.evidence_segment_ids or self.speech_evidence_ids or self.frame_evidence_ids)
        if not has_evidence and self.status == EvidenceStatus.supported:
            self.status = EvidenceStatus.unsupported
        self.title = self.title or self.topic
        self.summary = self.summary or self.claim
        if not self.support_modalities:
            modalities: list[EvidenceType] = []
            if self.evidence_segment_ids or self.speech_evidence_ids:
                modalities.append("speech")
            if self.frame_evidence_ids:
                modalities.append("frame")
            self.support_modalities = modalities
        else:
            self.support_modalities = [item for item in self.support_modalities if item in {"speech", "frame", "frame_caption"}]
        if not self.modality_support:
            self.modality_support = {
                "speech": 1.0 if (self.evidence_segment_ids or self.speech_evidence_ids) else 0.0,
                "vision": 1.0 if (self.frame_evidence_ids or self.frame_caption_evidence) else 0.0,
            }
        return self


class Storyline(BaseModel):
    video_id: str
    query: str
    nodes: list[StorylineNode]
    validation_warnings: list[str] = Field(default_factory=list)
    evidence_coverage: dict[str, Any] = Field(default_factory=dict)
    modality_usage_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvidenceSet(BaseModel):
    video_id: str
    query: str
    segment_ids: list[str]
    scores: dict[str, float] = Field(default_factory=dict)
    quotes: dict[str, str] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    evidence_id: str
    video_id: str
    segment_id: str
    evidence_type: EvidenceType
    text: str
    start: float
    end: float
    frame_id: str | None = None
    image_path: str | None = None
    timestamp: float | None = None
    score: float = 0.0
    raw_text: str | None = None
    normalized_text: str | None = None


class VideoFrame(BaseModel):
    frame_id: str
    timestamp: float
    path: str
    perceptual_hash: str | None = None
    embedding: list[float] | None = None
    selected: bool = False
    selection_reason: str | None = None


class VideoVisualProfile(BaseModel):
    video_id: str
    video_path: str
    motion_score: float = Field(ge=0.0, le=1.0)
    visual_complexity_score: float = Field(ge=0.0, le=1.0)
    color_variation_score: float = Field(ge=0.0, le=1.0)
    resolution_width: int | None = None
    resolution_height: int | None = None
    fps: float | None = None
    has_subtitles: bool = False
    visual_strength_score: float = Field(ge=0.0, le=1.0)
    visual_class: Literal["weak_visual", "strong_visual"]
    recommended_frame_fps: float
    recommended_max_frames_per_segment: int
    recommended_min_representative_frames: int
    reason: str = ""


class FrameCorrelationScore(BaseModel):
    frame_id: str
    timestamp: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class VideoCorrelationProfile(BaseModel):
    frame_confidences: list[float] = Field(default_factory=list)
    frame_scores: list[FrameCorrelationScore] = Field(default_factory=list)
    avg_confidence: float = Field(ge=0.0, le=1.0)
    frame_interval: float
    refine_frame_count: int
    representative_frame_strategy: Literal["random", "default"]
    relevance_level: Literal["low", "medium", "high", "text_only"]
    status: Literal["success", "video_unavailable", "text_too_short", "missing_model", "failed"]
    correlation_available: bool = True
    fallback_reason: str | None = None
    reason: str = ""


class FrameCaption(BaseModel):
    frame_id: str
    timestamp: float
    image_path: str | None = None
    caption: str
    model: str
    confidence: float | None = None
    status: str = "success"


class OCRResult(BaseModel):
    frame_id: str
    timestamp: float
    text: str
    confidence: float | None = None
    boxes: list[dict[str, float]] | None = None


class MultimodalSegment(BaseModel):
    segment_id: str
    start: float
    end: float
    transcript_text: str = ""
    # transcript_text remains the retrieval/display text for backwards compatibility.
    # raw_transcript_text is immutable source evidence; normalized_transcript_text is
    # the derived, retrieval-friendly representation.
    raw_transcript_text: str = ""
    normalized_transcript_text: str = ""
    normalization_applied: bool = False
    replacements: list[dict[str, Any]] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    representative_frame_ids: list[str] = Field(default_factory=list)
    visual_captions: list[FrameCaption] = Field(default_factory=list)
    frame_caption_status: str = "not_requested"
    ocr_text: str = ""
    visual_summary: str = ""
    modality_weight: dict[str, float] = Field(default_factory=lambda: {"text": 1.0, "vision": 0.0})
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)

    @field_validator("evidence_types", mode="before")
    @classmethod
    def filter_invalid_evidence_types(cls, value: object) -> list[str]:
        if not value:
            return []
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item) in {"speech", "frame", "frame_caption"}]

    @model_validator(mode="after")
    def fill_evidence_types(self) -> "MultimodalSegment":
        types: list[EvidenceType] = []
        if self.transcript_text.strip():
            types.append("speech")
        if self.representative_frame_ids:
            types.append("frame")
        if self.visual_summary.strip() or any(caption.caption.strip() for caption in self.visual_captions):
            types.append("frame_caption")
        if self.visual_captions:
            statuses = {caption.status for caption in self.visual_captions}
            if "success" in statuses:
                self.frame_caption_status = "success"
            elif "missing_model" in statuses:
                self.frame_caption_status = "missing_vision_model"
            elif "failed" in statuses:
                self.frame_caption_status = "failed"
        if not self.evidence_types:
            self.evidence_types = types
        else:
            self.evidence_types = [item for item in self.evidence_types if item in {"speech", "frame", "frame_caption"}]
        self.modality_weight = {
            "text": float(self.modality_weight.get("text", 0.0)),
            "vision": float(self.modality_weight.get("vision", 0.0)),
        }
        return self


class VisualEvidence(BaseModel):
    evidence_id: str
    frame_id: str
    timestamp: float
    evidence_type: Literal["frame", "frame_caption"]
    content: str
    confidence: float | None = None


class MultimodalEvidenceSet(BaseModel):
    video_id: str
    query: str
    segment_ids: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    quotes: dict[str, str] = Field(default_factory=dict)
    time_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)


class RefineRange(BaseModel):
    start: float
    end: float
    reason: str
    priority: int = 1

    @field_validator("end")
    @classmethod
    def refine_end_must_not_precede_start(cls, value: float, info: Any) -> float:
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("refine range end must be greater than or equal to start")
        return value


class EvidenceGapDecision(BaseModel):
    is_video_relevant: bool
    relevance_reason: str
    early_exit_reply: str | None = None
    should_refine: bool = False
    reasons: list[str] = Field(default_factory=list)
    target_ranges: list[RefineRange] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    round_traces: list[str] = Field(default_factory=list)


TaskType = Literal["factual", "summary", "explanation", "comparison", "evidence_check", "critique", "action_items", "procedure"]
TemporalScope = Literal["none", "point", "interval", "event_local", "whole_video"]
TemporalRelation = Literal["none", "before_after", "sequence", "stage", "transition"]
ContentModality = Literal["speech", "visual"]


class TimeConstraint(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    source_text: str

    @field_validator("end")
    @classmethod
    def time_constraint_end_must_not_precede_start(cls, value: float, info: Any) -> float:
        start = info.data.get("start")
        if start is not None and value < start:
            raise ValueError("time constraint end must be greater than or equal to start")
        return value


class RuleHints(BaseModel):
    possible_temporal_relation: TemporalRelation = "none"
    explicit_visual_reference: bool = False
    explicit_speech_reference: bool = False
    confidence: Literal["low", "medium", "high"] = "low"


class DeterministicQuestionContext(BaseModel):
    video_title: str = ""
    video_duration: float | None = None
    has_real_speech: bool = False
    has_audio: bool | None = None
    choice_based: bool = False
    explicit_whole_video: bool = False
    choices: list[str] = Field(default_factory=list)
    explicit_timestamps: list[TimeConstraint] = Field(default_factory=list)
    explicit_intervals: list[TimeConstraint] = Field(default_factory=list)
    transcript_preview: str = ""
    rule_hints: RuleHints = Field(default_factory=RuleHints)


class QuestionAnalysis(BaseModel):
    task_type: TaskType = "factual"
    temporal_scope: TemporalScope = "none"
    temporal_relation: TemporalRelation = "none"
    answer_target: str = ""
    content_modalities: list[ContentModality] = Field(default_factory=list)
    choice_based: bool = False
    reason: str = ""
    analysis_method: Literal["llm", "fallback"] = "fallback"

    @field_validator("content_modalities")
    @classmethod
    def unique_modalities(cls, value: list[ContentModality]) -> list[ContentModality]:
        return list(dict.fromkeys(value))


class RequiredFact(BaseModel):
    fact: str
    modality: ContentModality


class ClaimRequirement(BaseModel):
    id: str
    description: str
    required: bool = True
    support_requirement: Literal["direct", "derived", "either"] = "either"
    preferred_modalities: list[str] = Field(default_factory=list)
    importance: Literal["core", "supporting", "optional"] = "core"


class RelationRequirement(BaseModel):
    id: str
    subject_claim_id: str | None = None
    object_claim_id: str | None = None
    description: str
    relation_type: Literal[
        "before", "after", "causes", "explains", "compares", "differs_from",
        "part_of", "same_as", "other",
    ] = "other"
    required: bool = True


class ScopeRequirement(BaseModel):
    target: Literal["local", "relevant_evidence", "multi_event", "whole_video"] = "local"
    completeness: Literal["none", "best_effort", "strict"] = "none"
    allow_partial_answer: bool = True
    description: str | None = None


class ClaimSupportResult(BaseModel):
    claim_id: str
    claim: str
    status: Literal["supported", "partial", "unsupported", "contradicted", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    confidence: ConfidenceLevel | None = None


class RelationSupportResult(BaseModel):
    relation_id: str
    status: Literal["supported", "partial", "unsupported", "contradicted", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class ScopeAssessment(BaseModel):
    status: Literal["complete", "probably_complete", "incomplete", "unknown", "not_applicable"]
    requirement: ScopeRequirement
    evidence_coverage_reason: str = ""
    missing_scope: list[str] = Field(default_factory=list)


class EvidenceGateResult(BaseModel):
    status: Literal["fully_supported", "supported_with_scope_limit", "needs_refinement", "insufficient"]
    claim_results: list[ClaimSupportResult] = Field(default_factory=list)
    relation_results: list[RelationSupportResult] = Field(default_factory=list)
    scope_assessment: ScopeAssessment
    missing_requirements: list[str] = Field(default_factory=list)
    refinement_targets: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidencePlan(BaseModel):
    required_modalities: list[ContentModality] = Field(default_factory=list)
    required_facts: list[RequiredFact] = Field(default_factory=list)
    decision_facts: list[str] = Field(default_factory=list)
    min_distinct_timestamps: int = Field(default=1, ge=0)
    requires_temporal_order: bool = False
    requires_global_coverage: bool = False
    requires_visual_identity: bool = False
    human_readable_requirements: list[str] = Field(default_factory=list)
    reason: str = ""
    planning_method: Literal["llm", "fallback"] = "fallback"
    claims: list[ClaimRequirement] = Field(default_factory=list)
    relations: list[RelationRequirement] = Field(default_factory=list)
    scope: ScopeRequirement = Field(default_factory=ScopeRequirement)

    @field_validator("required_modalities")
    @classmethod
    def unique_required_modalities(cls, value: list[ContentModality]) -> list[ContentModality]:
        return list(dict.fromkeys(value))


class RefinementDiagnostics(BaseModel):
    """Structured diagnosis of a refinement attempt.

    Coverage describes where we looked; grounding describes whether the
    observations actually support the requested facts.  They are deliberately
    separate so broad sampling cannot masquerade as semantic proof.
    """

    sampling_coverage: Literal["sufficient", "partial", "insufficient", "not_applicable"] = "not_applicable"
    visual_fact_grounding: Literal["sufficient", "partial", "insufficient", "not_applicable"] = "not_applicable"
    temporal_fact_grounding: Literal["sufficient", "partial", "insufficient", "not_applicable"] = "not_applicable"
    missing_fact_types: list[str] = Field(default_factory=list)
    reason: str = ""


class RetrievalPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    time_constraints: list[TimeConstraint] = Field(default_factory=list)
    reason: str = ""


class ResolvedIntent(BaseModel):
    """Legacy compatibility view over QuestionAnalysis and EvidencePlan."""

    original_question: str
    rewritten_question: str
    question_type: str
    retrieval_queries: list[str] = Field(default_factory=list)
    required_evidence_types: list[EvidenceType] = Field(default_factory=list)
    optional_evidence_types: list[EvidenceType] = Field(default_factory=list)
    needs_visual_check: bool = False
    can_use_general_knowledge: bool = False
    general_knowledge_policy: str = "only_after_video_evidence_and_clearly_labeled"
    reason: str = ""
    temporal_scope: TemporalScope = "none"
    temporal_relation: TemporalRelation = "none"
    evidence_requirements: list[str] = Field(default_factory=list)
    distinguishing_facts: list[str] = Field(default_factory=list)
    min_distinct_visual_timestamps: int = 0
    resolution_method: str = "rules"
    deterministic_context: DeterministicQuestionContext | None = None
    question_analysis: QuestionAnalysis | None = None
    evidence_plan: EvidencePlan | None = None


class PartialPipelineRestartResult(BaseModel):
    ranges: list[RefineRange]
    added_frames: int = 0
    added_text_segments: int = 0
    added_captions: int = 0
    added_evidence_ids: list[str] = Field(default_factory=list)
    status: str = "unknown"
    refinement_strategy: str = "local_first"
    refined_frame_count: int = 0


class OutlineItem(BaseModel):
    timestamp: str
    topic: str
    key_points: list[str]
    segment_ids: list[str]


class AnalysisItem(BaseModel):
    claim: str
    evidence_segment_ids: list[str]
    caveats: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    action: str
    evidence_segment_ids: list[str]


class QuoteItem(BaseModel):
    quote: str
    segment_id: str
    timestamp: str


class SummaryReport(BaseModel):
    video_id: str
    quick_overview: list[str]
    structured_outline: list[OutlineItem]
    deep_analysis: list[AnalysisItem]
    action_items: list[ActionItem]
    important_quotes: list[QuoteItem]
    open_questions: list[str]
    modality_note: str
    evidence_coverage: dict[str, Any] = Field(default_factory=dict)
    confidence: ConfidenceLevel = ConfidenceLevel.medium
    generation_status: str = "unknown"
    summary_strategy: str = "single_pass"
    num_chunks: int = 1
    chunk_summary_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    canonical_entities: list[CanonicalEntity] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class QAAnswer(BaseModel):
    question: str
    answer: str
    evidence_segment_ids: list[str]
    timestamps: list[str]
    quotes: list[str]
    confidence: ConfidenceLevel
    needs_visual_check: bool
    reason: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    agreement_score: float = 0.0
    query_evidence_ids: list[str] = Field(default_factory=list)
    answer_evidence_ids: list[str] = Field(default_factory=list)
    dpp_used: bool = False
    dpp_details: dict[str, Any] = Field(default_factory=dict)
    suggested_followup_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_contract(self) -> "QAAnswer":
        if self.answer.strip() and "证据不足" not in self.answer and not self.evidence_segment_ids:
            raise ValueError("QAAnswer requires evidence_segment_ids unless evidence is insufficient")
        return self


class ConversationTurn(BaseModel):
    turn_id: str
    video_id: str
    conversation_id: str | None = None
    user_question: str
    answer: str
    current_video_answer: str = ""
    historical_evidence: list[dict[str, Any]] = Field(default_factory=list)
    background_knowledge: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    timestamps: list[str] = Field(default_factory=list)
    evidence_types: list[EvidenceType] = Field(default_factory=list)
    confidence: ConfidenceLevel
    agreement_score: float = 0.0
    query_evidence_ids: list[str] = Field(default_factory=list)
    answer_evidence_ids: list[str] = Field(default_factory=list)
    dpp_used: bool = False
    dpp_details: dict[str, Any] = Field(default_factory=dict)
    needs_visual_check: bool
    reason: str
    question_analysis: QuestionAnalysis | None = None
    evidence_plan: EvidencePlan | None = None
    structured_evidence_gate: dict[str, list[str]] = Field(
        default_factory=lambda: {"satisfied_constraints": [], "unmet_constraints": []}
    )
    fact_support: list[dict[str, Any]] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = True
    stop_reason: str = ""
    refinement_diagnostics: RefinementDiagnostics = Field(default_factory=RefinementDiagnostics)
    suggested_followup_questions: list[str] = Field(default_factory=list)
    candidate_claims: list[dict[str, Any]] = Field(default_factory=list)
    evidence_gate_result: EvidenceGateResult | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationSession(BaseModel):
    conversation_id: str
    video_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ModalityProfile(BaseModel):
    mode: ModalityMode
    speech_density: float = Field(ge=0.0)
    visual_required_score: float = Field(ge=0.0, le=1.0)
    ocr_required_score: float = Field(ge=0.0, le=1.0)
    reason: str
