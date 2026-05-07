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
    settings = Settings(_env_file=None, ASR_PROVIDER="faster-whisper", ASR_MODEL="small")
    adapter = get_asr_adapter(settings)
    assert isinstance(adapter, FasterWhisperASRAdapter)
