from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from quick_auditory_learning.costs import SUPPORTED_COMPLETION_MODEL_NAMES


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUICK_AUDITORY_LEARNING_", extra="ignore")

    data_dir: Path = Path("/workspace/_data/quick_auditory_learning")
    cache_dir: Path = Path("/workspace/_cache/quick-auditory-learning")
    log_dir: Path = Path("/workspace/_tmp/quick_auditory_learning/logs")
    postgres_dsn: str = "postgresql://quick_auditory_learning:quick_auditory_learning@db:5432/quick_auditory_learning"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QUICK_AUDITORY_LEARNING_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    jsonl_path: Path | None = None
    embedding_model_name: str = "text-embedding-3-large"
    llm_model: str = Field(
        default="gpt-5.4-nano",
        validation_alias=AliasChoices("QUICK_AUDITORY_LEARNING_LLM_MODEL"),
    )
    frontend_url: str | None = None
    voicevox_url: str = Field(
        default="http://voicevox:50021",
        validation_alias=AliasChoices("VOICEVOX_URL", "QUICK_AUDITORY_LEARNING_VOICEVOX_URL"),
    )
    voicevox_speaker_id: str = Field(
        default="1",
        validation_alias=AliasChoices("VOICEVOX_SPEAKER_ID", "QUICK_AUDITORY_LEARNING_VOICEVOX_SPEAKER_ID"),
    )
    voicevox_speed_scale: float = Field(
        default=1.25,
        validation_alias=AliasChoices("VOICEVOX_SPEED_SCALE", "QUICK_AUDITORY_LEARNING_VOICEVOX_SPEED_SCALE"),
    )
    voicevox_volume_scale: float = Field(
        default=1.0,
        validation_alias=AliasChoices("VOICEVOX_VOLUME_SCALE", "QUICK_AUDITORY_LEARNING_VOICEVOX_VOLUME_SCALE"),
    )

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model(cls, value: str) -> str:
        if value not in SUPPORTED_COMPLETION_MODEL_NAMES:
            supported = ", ".join(sorted(SUPPORTED_COMPLETION_MODEL_NAMES))
            raise ValueError(f"QUICK_AUDITORY_LEARNING_LLM_MODEL must be one of: {supported}")
        return value


settings = Settings()
