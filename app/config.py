from functools import lru_cache
import logging
import os
from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class Settings(BaseSettings):
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("LLM_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY"))
    openai_base_url: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_API_BASE", "API_BASE_URL"))
    llm_provider: str = Field(default="zhipu", alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, validation_alias=AliasChoices("LLM_MODEL", "CHAT_MODEL", "SUMMARY_MODEL"))
    llm_base_url: str = Field(
        default=DEFAULT_RUNTIME_BASE_URL,
        validation_alias=AliasChoices("LLM_BASE_URL", "BIGMODEL_BASE_URL", "ZHIPU_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "API_BASE_URL"),
    )
    llm_summary_model: str = Field(default="glm-4.6", validation_alias=AliasChoices("LLM_SUMMARY_MODEL", "SUMMARY_MODEL", "LLM_MODEL", "CHAT_MODEL"))
    storyline_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_storyline_API_KEY", "OPENAI_STORYLINE_API_KEY"))
    storyline_base_url: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_storyline_BASE_URL", "OPENAI_STORYLINE_BASE_URL", "OPENAI_BASE_URL"))
    storyline_model: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_storyline_MODEL", "OPENAI_STORYLINE_MODEL", "LLM_MODEL"))
    embedding_base_url: str = Field(default=DEFAULT_RUNTIME_BASE_URL, validation_alias=AliasChoices("EMBEDDING_BASE_URL", "BIGMODEL_BASE_URL", "ZHIPU_BASE_URL"))
    embedding_model: str | None = Field(default=None, alias="EMBEDDING_MODEL")
    obsidian_vault_path: Path | None = Field(default=None, alias="OBSIDIAN_VAULT_PATH")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    asr_model: str = Field(default="small", alias="ASR_MODEL")
    vision_api_key: str | None = Field(default=None, validation_alias=AliasChoices("VLM_API_KEY", "VISION_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY"))
    vision_base_url: str | None = Field(default=None, validation_alias=AliasChoices("VISION_BASE_URL", "VLM_BASE_URL"))
    vision_model: str | None = Field(default=None, validation_alias=AliasChoices("VISION_MODEL", "VLM_MODEL", "VLM_CAPTION_MODEL"))
    vlm_provider: str = Field(default="zhipu", alias="VLM_PROVIDER")
    vlm_base_url: str | None = Field(default=None, validation_alias=AliasChoices("VLM_BASE_URL", "VISION_BASE_URL", "BIGMODEL_BASE_URL", "ZHIPU_BASE_URL"))
    vlm_caption_model: str | None = Field(default=None, validation_alias=AliasChoices("VLM_CAPTION_MODEL", "VLM_MODEL", "VISION_MODEL"))
    llm_correlation_provider: str = Field(default="zhipu", validation_alias=AliasChoices("LLM_correlation_PROVIDER", "LLM_CORRELATION_PROVIDER"))
    llm_correlation_api_key: str | None = Field(default=None, validation_alias=AliasChoices("LLM_correlation_API_KEY", "LLM_CORRELATION_API_KEY", "ZHIPU_API_KEY", "VLM_API_KEY", "OPENAI_API_KEY"))
    llm_correlation_base_url: str | None = Field(default=None, validation_alias=AliasChoices("LLM_correlation_BASE_URL", "LLM_CORRELATION_BASE_URL", "LLM_BASE_URL", "VLM_BASE_URL"))
    llm_correlation_model: str | None = Field(default=None, validation_alias=AliasChoices("LLM_correlation_MODEL", "LLM_CORRELATION_MODEL", "LLM_SUMMARY_MODEL", "VLM_CAPTION_MODEL"))
    evidence_gap_agent_mode: str = Field(default="auto", validation_alias=AliasChoices("EVIDENCE_GAP_AGENT_MODE", "evidence_GAP_AGENT_MODE"))
    evidence_llm_provider: str = Field(default="zhipu", validation_alias=AliasChoices("EVIDENCE_LLM_CORRELATION_PROVIDER", "evidence_LLM_correlation_PROVIDER"))
    evidence_llm_api_key: str | None = Field(default=None, validation_alias=AliasChoices("EVIDENCE_LLM_CORRELATION_API_KEY", "evidence_LLM_correlation_API_KEY"))
    evidence_llm_base_url: str | None = Field(default=None, validation_alias=AliasChoices("EVIDENCE_LLM_BASE_URL", "evidence_LLM_BASE_URL"))
    evidence_llm_model: str | None = Field(default=None, validation_alias=AliasChoices("EVIDENCE_LLM_SUMMARY_MODEL", "evidence_LLM_SUMMARY_MODEL"))
    ocr_provider: str = Field(default="none", alias="OCR_PROVIDER")
    ocr_engine: str = Field(default="none", alias="OCR_ENGINE")
    ocr_lang: str = Field(default="eng", alias="OCR_LANG")
    ocr_api_key: str | None = Field(default=None, alias="OCR_API_KEY")
    ocr_base_url: str | None = Field(default=None, alias="OCR_BASE_URL")
    ocr_model: str | None = Field(default=None, alias="OCR_MODEL")
    asr_provider: str = Field(default="local", alias="ASR_PROVIDER")
    asr_api_key: str | None = Field(default=None, alias="ASR_API_KEY")
    asr_base_url: str | None = Field(default=None, alias="ASR_BASE_URL")
    vector_store: str = Field(default="memory", alias="VECTOR_STORE")
    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
    vector_top_k: int = Field(default=24, alias="VECTOR_TOP_K")
    bm25_top_k: int = Field(default=24, alias="BM25_TOP_K")
    rrf_k: int = Field(default=60, validation_alias=AliasChoices("RRF_K", "RRQ_K"))
    vector_rrf_weight: float = Field(default=1.0, validation_alias=AliasChoices("VECTOR_RRF_WEIGHT", "VECTOR_RRQ_WEIGHT"))
    bm25_rrf_weight: float = Field(default=1.0, validation_alias=AliasChoices("BM25_RRF_WEIGHT", "BM25_RRQ_WEIGHT"))
    chroma_path: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PATH")
    ytdlp_cookies_file: Path | None = Field(default=None, alias="YTDLP_COOKIES_FILE")
    ytdlp_cookies_from_browser: str | None = Field(default=None, alias="YTDLP_COOKIES_FROM_BROWSER")
    mempalace_provider: str = Field(default="noop", alias="MEMPALACE_PROVIDER")
    mempalace_endpoint: str | None = Field(default=None, alias="MEMPALACE_ENDPOINT")
    mempalace_api_key: str | None = Field(default=None, alias="MEMPALACE_API_KEY")
    mempalace_command: str = Field(default="python", alias="MEMPALACE_COMMAND")
    mempalace_args: str = Field(default="-m mempalace.mcp_server", alias="MEMPALACE_ARGS")
    mempalace_palace_path: Path | None = Field(default=None, alias="MEMPALACE_PALACE_PATH")
    mempalace_timeout_seconds: float = Field(default=30.0, alias="MEMPALACE_TIMEOUT_SECONDS")
    mempalace_auto_status: bool = Field(default=True, alias="MEMPALACE_AUTO_STATUS")
    memory_promotion_enabled: bool = Field(default=True, alias="MEMORY_PROMOTION_ENABLED")
    request_timeout_seconds: float = Field(default=240.0, alias="REQUEST_TIMEOUT_SECONDS")
    pipeline_stage_restart_limit: int = Field(default=2, alias="PIPELINE_STAGE_RESTART_LIMIT")
    summary_max_segments: int = Field(default=24, alias="SUMMARY_MAX_SEGMENTS")
    summary_segment_chars: int = Field(default=500, alias="SUMMARY_SEGMENT_CHARS")
    summary_caption_chars: int = Field(default=300, alias="SUMMARY_CAPTION_CHARS")
    summary_map_concurrency: int = Field(default=1, alias="SUMMARY_MAP_CONCURRENCY")
    storyline_top_k: int = Field(default=8, alias="STORYLINE_TOP_K")
    vlm_caption_concurrency: int = Field(default=3, alias="VLM_CAPTION_CONCURRENCY")
    vlm_caption_retries: int = Field(default=2, alias="VLM_CAPTION_RETRIES")
    visual_classifier_enabled: bool = Field(default=True, alias="VISUAL_CLASSIFIER_ENABLED")
    visual_strength_threshold: float = Field(default=0.48, alias="VISUAL_STRENGTH_THRESHOLD")
    weak_visual_frame_fps: float = Field(default=0.2, alias="WEAK_VISUAL_FRAME_FPS")
    strong_visual_frame_fps: float = Field(default=1.0, alias="STRONG_VISUAL_FRAME_FPS")
    weak_visual_max_frames_per_segment: int = Field(default=1, alias="WEAK_VISUAL_MAX_FRAMES_PER_SEGMENT")
    strong_visual_max_frames_per_segment: int = Field(default=3, alias="STRONG_VISUAL_MAX_FRAMES_PER_SEGMENT")
    correlation_text_min_chars: int = Field(default=80, alias="CORRELATION_TEXT_MIN_CHARS")
    correlation_sample_frames: int = Field(default=10, alias="CORRELATION_SAMPLE_FRAMES")
    correlation_low_threshold: float = Field(default=0.2, alias="CORRELATION_LOW_THRESHOLD")
    correlation_high_threshold: float = Field(default=0.8, alias="CORRELATION_HIGH_THRESHOLD")
    llm_max_concurrency: int = Field(default=1, alias="LLM_MAX_CONCURRENCY")
    llm_min_interval_seconds: float = Field(default=3.0, alias="LLM_MIN_INTERVAL_SECONDS")
    vlm_min_interval_seconds: float = Field(default=2.0, alias="VLM_MIN_INTERVAL_SECONDS")
    embedding_max_concurrency: int = Field(default=1, alias="EMBEDDING_MAX_CONCURRENCY")
    embedding_min_interval_seconds: float = Field(default=1.0, alias="EMBEDDING_MIN_INTERVAL_SECONDS")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_path: Path | None = Field(default=None, alias="EMBEDDING_CACHE_PATH")
    ytdlp_max_concurrency: int = Field(default=1, alias="YTDLP_MAX_CONCURRENCY")
    ytdlp_min_interval_seconds: float = Field(default=6.0, alias="YTDLP_MIN_INTERVAL_SECONDS")
    asr_max_concurrency: int = Field(default=1, alias="ASR_MAX_CONCURRENCY")
    asr_min_interval_seconds: float = Field(default=2.0, alias="ASR_MIN_INTERVAL_SECONDS")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    api_retry_base_delay_seconds: float = Field(default=2.0, alias="API_RETRY_BASE_DELAY_SECONDS")
    api_rate_limit_cooldown_seconds: float = Field(default=30.0, alias="API_RATE_LIMIT_COOLDOWN_SECONDS")
    pipeline_resume: bool = Field(default=True, alias="PIPELINE_RESUME")
    pipeline_force_stage: str | None = Field(default=None, alias="PIPELINE_FORCE_STAGE")
    allow_coding_endpoint_for_runtime: bool = Field(default=False, alias="ALLOW_CODING_ENDPOINT_FOR_RUNTIME")
    strict_llm: bool = Field(default=False, alias="STRICT_LLM")
    strict_runtime: bool = Field(default=True, alias="STRICT_RUNTIME")
    force_refresh: bool = Field(default=False, alias="FORCE_REFRESH")
    enable_ocr: bool = Field(default=False, alias="ENABLE_OCR")
    evidence_types_config: str = Field(default="speech,frame,frame_caption", alias="EVIDENCE_TYPES")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    @model_validator(mode="after")
    def normalize_runtime_secrets(self) -> "Settings":
        if not _usable_secret(self.openai_api_key):
            self.openai_api_key = _first_usable_env("ZHIPU_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        if not _usable_secret(self.storyline_api_key):
            self.storyline_api_key = _first_usable_env("OPENAI_storyline_API_KEY", "OPENAI_STORYLINE_API_KEY")
        if not _usable_secret(self.vision_api_key):
            self.vision_api_key = _first_usable_env("VISION_API_KEY", "VLM_API_KEY", "ZHIPU_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")
        if not _usable_secret(self.llm_correlation_api_key):
            self.llm_correlation_api_key = _first_usable_env("LLM_correlation_API_KEY", "LLM_CORRELATION_API_KEY", "ZHIPU_API_KEY", "VLM_API_KEY", "OPENAI_API_KEY")
        if not _usable_secret(self.evidence_llm_api_key):
            self.evidence_llm_api_key = _first_usable_env("evidence_LLM_correlation_API_KEY", "EVIDENCE_LLM_CORRELATION_API_KEY")
        return self

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def has_llm_config(self) -> bool:
        return bool(self.openai_api_key and self.runtime_llm_base_url and self.runtime_llm_model)

    @property
    def has_storyline_llm_config(self) -> bool:
        return bool(self.storyline_api_key and self.runtime_storyline_base_url and self.runtime_storyline_model)

    @property
    def has_embedding_config(self) -> bool:
        return bool(self.openai_api_key and self.runtime_embedding_base_url and self.embedding_model)

    @property
    def has_vision_config(self) -> bool:
        return bool(self.vision_api_key and self.runtime_vlm_base_url and self.runtime_vlm_model)

    @property
    def has_correlation_config(self) -> bool:
        return bool(self.llm_correlation_api_key and self.runtime_correlation_base_url and self.runtime_correlation_model)

    @property
    def has_evidence_llm_config(self) -> bool:
        return bool(self.evidence_llm_api_key and self.runtime_evidence_llm_base_url and self.runtime_evidence_llm_model)

    @property
    def has_cloud_asr_config(self) -> bool:
        return bool(self.asr_provider.lower() not in {"local", "faster-whisper", "faster_whisper", "whisper"} and self.asr_api_key and self.asr_base_url and self.asr_model)

    @property
    def has_cloud_ocr_config(self) -> bool:
        return bool(self.ocr_provider not in {"none", "local"} and self.ocr_api_key and self.ocr_base_url)

    @property
    def runtime_llm_base_url(self) -> str:
        return normalize_openai_base_url(self.llm_base_url or DEFAULT_RUNTIME_BASE_URL)

    @property
    def runtime_llm_model(self) -> str:
        return self.llm_summary_model or self.llm_model or "glm-4.6"

    @property
    def runtime_storyline_base_url(self) -> str:
        return normalize_openai_base_url(self.storyline_base_url or self.runtime_llm_base_url)

    @property
    def runtime_storyline_model(self) -> str:
        return self.storyline_model or self.llm_model or self.runtime_llm_model

    @property
    def runtime_embedding_base_url(self) -> str:
        return normalize_openai_base_url(self.embedding_base_url or DEFAULT_RUNTIME_BASE_URL)

    @property
    def runtime_vlm_base_url(self) -> str:
        return normalize_openai_base_url(self.vlm_base_url or self.vision_base_url or DEFAULT_RUNTIME_BASE_URL)

    @property
    def runtime_vlm_model(self) -> str | None:
        return self.vlm_caption_model or self.vision_model

    @property
    def runtime_correlation_base_url(self) -> str:
        return normalize_openai_base_url(self.llm_correlation_base_url or self.runtime_vlm_base_url or DEFAULT_RUNTIME_BASE_URL)

    @property
    def runtime_correlation_model(self) -> str | None:
        return self.llm_correlation_model or self.runtime_vlm_model or self.runtime_llm_model

    @property
    def runtime_evidence_llm_base_url(self) -> str:
        return normalize_openai_base_url(self.evidence_llm_base_url or self.runtime_correlation_base_url or DEFAULT_RUNTIME_BASE_URL)

    @property
    def runtime_evidence_llm_model(self) -> str | None:
        return self.evidence_llm_model or self.runtime_correlation_model or self.runtime_llm_model

    def runtime_config_summary(self) -> dict[str, str | None | bool]:
        return {
            "LLM_BASE_URL": self.runtime_llm_base_url,
            "LLM_SUMMARY_MODEL": self.runtime_llm_model,
            "LLM_PROVIDER": self.llm_provider,
            "STORYLINE_BASE_URL": self.runtime_storyline_base_url,
            "STORYLINE_MODEL": self.runtime_storyline_model,
            "HAS_STORYLINE_LLM_CONFIG": self.has_storyline_llm_config,
            "VLM_BASE_URL": self.runtime_vlm_base_url,
            "VLM_CAPTION_MODEL": self.runtime_vlm_model,
            "VLM_PROVIDER": self.vlm_provider,
            "LLM_correlation_PROVIDER": self.llm_correlation_provider,
            "LLM_correlation_BASE_URL": self.runtime_correlation_base_url,
            "LLM_correlation_MODEL": self.runtime_correlation_model,
            "HAS_CORRELATION_CONFIG": self.has_correlation_config,
            "EVIDENCE_GAP_AGENT_MODE": self.evidence_gap_agent_mode,
            "EVIDENCE_LLM_BASE_URL": self.runtime_evidence_llm_base_url,
            "EVIDENCE_LLM_SUMMARY_MODEL": self.runtime_evidence_llm_model,
            "HAS_EVIDENCE_LLM_CONFIG": self.has_evidence_llm_config,
            "EMBEDDING_BASE_URL": self.runtime_embedding_base_url,
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_CACHE_ENABLED": self.embedding_cache_enabled,
            "STRICT_RUNTIME": self.strict_runtime,
            "STRICT_LLM": self.strict_llm,
            "FORCE_REFRESH": self.force_refresh,
            "REQUEST_TIMEOUT_SECONDS": self.request_timeout_seconds,
            "SUMMARY_MAX_SEGMENTS": self.summary_max_segments,
            "SUMMARY_SEGMENT_CHARS": self.summary_segment_chars,
            "SUMMARY_CAPTION_CHARS": self.summary_caption_chars,
            "SUMMARY_MAP_CONCURRENCY": self.summary_map_concurrency,
            "STORYLINE_TOP_K": self.storyline_top_k,
            "VLM_CAPTION_CONCURRENCY": self.vlm_caption_concurrency,
            "VLM_CAPTION_RETRIES": self.vlm_caption_retries,
            "VISUAL_CLASSIFIER_ENABLED": self.visual_classifier_enabled,
            "VISUAL_STRENGTH_THRESHOLD": self.visual_strength_threshold,
            "RETRIEVAL_MODE": self.retrieval_mode,
            "VECTOR_TOP_K": str(self.vector_top_k),
            "BM25_TOP_K": str(self.bm25_top_k),
            "RRF_K": str(self.rrf_k),
            "LLM_MAX_CONCURRENCY": self.llm_max_concurrency,
            "VLM_CAPTION_CONCURRENCY": self.vlm_caption_concurrency,
            "YTDLP_MAX_CONCURRENCY": self.ytdlp_max_concurrency,
            "YTDLP_COOKIES_FROM_BROWSER": self.ytdlp_cookies_from_browser,
            "PIPELINE_RESUME": self.pipeline_resume,
            "PIPELINE_FORCE_STAGE": self.pipeline_force_stage,
            "ALLOW_CODING_ENDPOINT_FOR_RUNTIME": self.allow_coding_endpoint_for_runtime,
            "MEMORY_PROMOTION_ENABLED": self.memory_promotion_enabled,
        }

    def validate_runtime_config(self) -> None:
        base_url = self.runtime_llm_base_url.lower()
        if "/coding/paas/v4" in base_url:
            message = (
                "Runtime LLM summary is using coding endpoint; this is likely wrong. "
                "Use https://open.bigmodel.cn/api/paas/v4 for normal summary tasks."
            )
            logger.warning(message)
        if "/coding/paas/v4" in base_url and self.strict_llm and not self.allow_coding_endpoint_for_runtime:
            raise RuntimeError(
                "Invalid runtime LLM endpoint: LLM_BASE_URL must not contain /coding/paas/v4. "
                "Set LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4 or explicitly set "
                "ALLOW_CODING_ENDPOINT_FOR_RUNTIME=true."
            )

    def log_runtime_config(self) -> None:
        self.validate_runtime_config()
        logger.info("Runtime config: %s", self.runtime_config_summary())


def normalize_openai_base_url(value: str) -> str:
    base = value.strip().rstrip("/")
    for suffix in ("/chat/completions", "/embeddings"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


def _usable_secret(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped and not (stripped.startswith("${") and stripped.endswith("}")))


def _first_usable_env(*keys: str) -> str | None:
    dotenv_values: dict[str, str | None] = {}
    try:
        from dotenv import dotenv_values as read_dotenv_values

        dotenv_values = read_dotenv_values(".env")
    except Exception:
        dotenv_values = {}
    for key in keys:
        value = os.environ.get(key)
        if _usable_secret(value):
            return value
        value = dotenv_values.get(key)
        if _usable_secret(value):
            return value
    return None


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    return settings
