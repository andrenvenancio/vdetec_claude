from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "change-me"
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://vdetec:vdetec@localhost:5432/vdetec"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"

    # Storage
    storage_backend: Literal["local", "s3"] = "local"
    media_root: str = "./media"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""

    # Computer Vision
    model_path: str = "./models/detector.pt"
    detector_conf_threshold: float = Field(0.45, ge=0.0, le=1.0)
    detector_iou_threshold: float = Field(0.50, ge=0.0, le=1.0)
    device: str = "cuda"
    batch_size: int = 8

    # Cameras
    camera_poll_interval_sec: int = 30
    rtsp_timeout_sec: int = 10

    # Alerts
    alert_stockout_threshold: int = 0
    alert_email_from: str = "noreply@vdetec.com"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # API
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
