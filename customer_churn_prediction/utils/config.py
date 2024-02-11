"""Pydantic based config class."""

from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Log level validation class."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AppConfig(BaseSettings):
    """Configuration settings, sourced from the environment."""

    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Standard out log level.",
    )

    model_config = SettingsConfigDict(env_prefix="CCP_")


app_config = AppConfig()  # type: ignore
