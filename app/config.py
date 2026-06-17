"""
TR: Tum uygulama ayarlari tek bir yerde, tip guvenli sekilde toplanir.
    `.env` dosyasindan ya da ortam degiskenlerinden okunur.
EN: All application settings are collected here in one type-safe place.
    Values are read from the `.env` file or environment variables.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    """TR: Desteklenen model saglayicilari. / EN: Supported model providers."""

    OPENAI = "openai"
    AZURE = "azure"
    COMPATIBLE = "compatible"  # vLLM, Triton, Together AI, Ollama, OpenRouter, ...


class ApiMode(str, Enum):
    """TR: OpenAI API sekli. / EN: OpenAI API shape."""

    RESPONSES = "responses"
    CHAT_COMPLETIONS = "chat_completions"


class SessionBackend(str, Enum):
    """TR: Konusma gecmisi deposu. / EN: Conversation history store."""

    MEMORY = "memory"
    SQLITE = "sqlite"
    REDIS = "redis"


class Settings(BaseSettings):
    """
    TR: Uygulamanin merkezi ayar nesnesi.
    EN: The application's central settings object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- TR: Genel / EN: General -------------------------------------------------
    model_provider: ModelProvider = ModelProvider.OPENAI
    agent_model: str = "gpt-5.5"
    log_level: str = "INFO"

    # --- TR: OpenAI / EN: OpenAI -------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # --- TR: OpenAI-compatible (vLLM/Triton/Together) / EN: OpenAI-compatible ----
    compatible_base_url: str = "http://localhost:8000/v1"
    compatible_api_key: str = "EMPTY"
    compatible_api_mode: ApiMode = ApiMode.CHAT_COMPLETIONS

    # --- TR: Azure OpenAI / Azure AI Foundry / EN: Azure -------------------------
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str | None = None

    # --- TR: Tracing / EN: Tracing ----------------------------------------------
    tracing_disabled: bool = False
    tracing_export_api_key: str | None = None
    workflow_name: str = "agent-boilerplate"

    # --- TR: Session / EN: Session ----------------------------------------------
    session_backend: SessionBackend = SessionBackend.MEMORY
    session_db_path: str = "conversations.db"
    redis_url: str = Field(default="redis://localhost:6379/0")

    @property
    def effective_model(self) -> str:
        """
        TR: Calismada kullanilacak model adi.
            Azure'da agent modeli "deployment" adidir.
        EN: The model name used at runtime.
            On Azure, the agent model is the "deployment" name.
        """
        if self.model_provider is ModelProvider.AZURE and self.azure_openai_deployment:
            return self.azure_openai_deployment
        return self.agent_model


@lru_cache
def get_settings() -> Settings:
    """
    TR: Ayarlari bir kez yukleyip onbellekler (singleton gibi davranir).
    EN: Loads settings once and caches them (behaves like a singleton).
    """
    return Settings()
