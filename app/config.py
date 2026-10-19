from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    public_base_url: str = ""
    redis_url: str = ""
    require_redis: bool = False
    vllm_base_url: str = ""
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    require_translator: bool = False
    allow_fake_models: bool = False
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    model_dir: str = ""
    max_live_sessions: int = 4
    translate_timeout_seconds: float = 3.5
    database_url: str = ""
    test_database_url: str = ""
    admin_username: str = ""
    admin_password: str = ""
    google_api_key: str = ""
    google_cloud_project: str = ""
    google_application_credentials: str = ""
    google_speech_region: str = "us"
    google_speech_model: str = "chirp_3"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    tts_voice_en: str = "en-US-Wavenet-D"
    tts_voice_id: str = "id-ID-Wavenet-B"
    rtmp_public_url: str = ""
    mediamtx_rtmp_url: str = ""
    rtmp_auth_secret: str = ""
    rtmp_read_password: str = ""
