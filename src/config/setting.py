from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    artifacts_dir: str = "artifacts"
    model_dir: str = "artifacts/model"
    feature_pipeline_dir: str = "artifacts/feature_pipeline"

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


settings = Settings()