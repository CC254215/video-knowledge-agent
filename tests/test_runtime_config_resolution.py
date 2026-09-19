from app.config import Settings
from app.transcript.asr import FasterWhisperASRAdapter, get_asr_adapter


def test_placeholder_api_keys_do_not_override_real_keys(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "real-zhipu-key")
    monkeypatch.setenv("VISION_API_KEY", "real-zhipu-key")
    settings = Settings(_env_file=None, LLM_API_KEY="${placeholder}", VLM_API_KEY="${placeholder}")
    assert settings.openai_api_key == "real-zhipu-key"
    assert settings.vision_api_key == "real-zhipu-key"
    assert settings.has_llm_config is True
    assert settings.has_vision_config is False  # no VLM model configured in this test


def test_faster_whisper_provider_uses_local_adapter():
    settings = Settings(
        _env_file=None,
        ASR_PROVIDER="faster-whisper",
        ASR_MODEL="small",
        ASR_DEVICE="cpu",
        ASR_COMPUTE_TYPE="int8",
        ASR_BEAM_SIZE=1,
        ASR_VAD_FILTER=True,
    )
    adapter = get_asr_adapter(settings)
    assert isinstance(adapter, FasterWhisperASRAdapter)
    assert adapter.device == "cpu"
    assert adapter.compute_type == "int8"
    assert adapter.beam_size == 1
    assert adapter.vad_filter is True


def test_faster_whisper_provider_accepts_gpu_configuration():
    settings = Settings(
        _env_file=None,
        ASR_PROVIDER="faster-whisper",
        ASR_MODEL="small",
        ASR_DEVICE="cuda",
        ASR_COMPUTE_TYPE="float16",
    )
    adapter = get_asr_adapter(settings)
    assert isinstance(adapter, FasterWhisperASRAdapter)
    assert adapter.device == "cuda"
    assert adapter.compute_type == "float16"
