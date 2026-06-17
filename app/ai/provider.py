"""
TR: SDK'nin "plug-and-play" kalbi burasi.
    Tek bir fonksiyon (`configure_provider`) ile projeyi OpenAI, Azure OpenAI ya da
    herhangi bir OpenAI-uyumlu backend (vLLM, NVIDIA Triton, Together AI, Ollama...)
    arasinda gecirebilirsiniz. Agent ve tool kodlariniz HIC degismez.

EN: This is the "plug-and-play" heart of the setup.
    A single function (`configure_provider`) lets you switch the project between
    OpenAI, Azure OpenAI, or any OpenAI-compatible backend (vLLM, NVIDIA Triton,
    Together AI, Ollama...). Your agent and tool code does NOT change at all.

TR: Senaryo: MVP'yi OpenAI ile yaparsiniz. Daha sonra modeli kendi GPU'nuzda
    Triton/vLLM uzerinde, OpenAI-uyumlu bir API ile ayaga kaldirirsiniz. Tek yapmaniz
    gereken `MODEL_PROVIDER=compatible` ve `COMPATIBLE_BASE_URL` ayarlamak.
EN: Scenario: you build the MVP on OpenAI. Later you serve the model on your own GPU
    via Triton/vLLM behind an OpenAI-compatible API. All you do is set
    `MODEL_PROVIDER=compatible` and `COMPATIBLE_BASE_URL`.
"""

from __future__ import annotations

import logging

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    set_tracing_export_api_key,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.config import ApiMode, ModelProvider, Settings

logger = logging.getLogger(__name__)

# TR: Saglayiciyi iki kez yapilandirmamak icin bayrak. / EN: Flag to avoid configuring twice.
_configured = False


def configure_provider(settings: Settings) -> None:
    """
    TR: Uygulama acilisinda BIR KEZ cagrilir. Global OpenAI client'ini ve tracing'i ayarlar.
    EN: Called ONCE at application startup. Configures the global OpenAI client and tracing.
    """
    global _configured
    if _configured:
        return

    if settings.model_provider is ModelProvider.OPENAI:
        _configure_openai(settings)
    elif settings.model_provider is ModelProvider.AZURE:
        _configure_azure(settings)
    elif settings.model_provider is ModelProvider.COMPATIBLE:
        _configure_compatible(settings)
    else:  # pragma: no cover - enum exhaustively handled above
        raise ValueError(f"Unknown provider: {settings.model_provider}")

    _configure_tracing(settings)
    _configured = True
    logger.info("Provider configured: %s (model=%s)", settings.model_provider.value, settings.effective_model)


def _configure_openai(settings: Settings) -> None:
    """
    TR: Standart OpenAI. base_url verilirse bir gateway/proxy uzerinden de gidebilir.
    EN: Standard OpenAI. If base_url is provided, traffic can route via a gateway/proxy.
    """
    if settings.openai_base_url:
        # TR: Ozel endpoint icin client'i biz olusturup global yapariz.
        # EN: For a custom endpoint we build the client ourselves and set it globally.
        client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        set_default_openai_client(client)
    # TR: Aksi halde SDK, OPENAI_API_KEY ortam degiskeniyle kendi client'ini olusturur.
    # EN: Otherwise the SDK creates its own client from the OPENAI_API_KEY env var.
    # TR: OpenAI, Responses API'yi destekler -> varsayilani degistirmeye gerek yok.
    # EN: OpenAI supports the Responses API -> no need to change the default.


def _configure_azure(settings: Settings) -> None:
    """
    TR: Azure OpenAI / Azure AI Foundry. Azure'a ozel `AsyncAzureOpenAI` client'i kullanilir.
        SDK bu client'i global olarak benimser; agentlar `deployment` adini model olarak alir.
    EN: Azure OpenAI / Azure AI Foundry. Uses the Azure-specific `AsyncAzureOpenAI` client.
        The SDK adopts it globally; agents use the `deployment` name as their model.
    """
    if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
        raise ValueError("Azure requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")

    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    # TR: Azure client'ini tracing icin kullanmiyoruz (anahtar OpenAI platformuna ait degil).
    # EN: Do not use the Azure client for tracing (its key is not for the OpenAI platform).
    set_default_openai_client(client, use_for_tracing=False)
    # TR: Azure deployment'lari genelde Chat Completions ile en uyumlu calisir.
    # EN: Azure deployments are generally most compatible with Chat Completions.
    set_default_openai_api("chat_completions")


def _configure_compatible(settings: Settings) -> None:
    """
    TR: OpenAI-uyumlu herhangi bir backend: vLLM, NVIDIA Triton (vLLM backend),
        Together AI, Ollama, LM Studio, OpenRouter, vb.
    EN: Any OpenAI-compatible backend: vLLM, NVIDIA Triton (vLLM backend),
        Together AI, Ollama, LM Studio, OpenRouter, etc.
    """
    client = AsyncOpenAI(
        base_url=settings.compatible_base_url,
        api_key=settings.compatible_api_key,  # TR: vLLM'de cogu zaman "EMPTY". / EN: Often "EMPTY" for vLLM.
    )
    # TR: Bu client'in anahtari OpenAI'a ait olmadigi icin tracing icin kullanmayiz.
    # EN: This client's key is not an OpenAI key, so we don't use it for tracing.
    set_default_openai_client(client, use_for_tracing=False)

    # TR: Bu backendlerin cogu Responses API'yi DESTEKLEMEZ; Chat Completions'a geceriz.
    # EN: Most of these backends do NOT support the Responses API; switch to Chat Completions.
    if settings.compatible_api_mode is ApiMode.CHAT_COMPLETIONS:
        set_default_openai_api("chat_completions")


def _configure_tracing(settings: Settings) -> None:
    """
    TR: Tracing yapilandirmasi. OpenAI disi saglayicilarda ya kapatilir ya da
        sadece trace yuklemek icin ayri bir OpenAI anahtari verilir.
    EN: Tracing configuration. For non-OpenAI providers, either disable it or
        provide a separate OpenAI key used only for uploading traces.
    """
    if settings.tracing_disabled:
        set_tracing_disabled(True)
        return

    if settings.tracing_export_api_key:
        set_tracing_export_api_key(settings.tracing_export_api_key)
    elif settings.model_provider is not ModelProvider.OPENAI:
        # TR: OpenAI disi saglayici + trace anahtari yok -> 401 yememek icin kapat.
        # EN: Non-OpenAI provider + no trace key -> disable to avoid 401 errors.
        logger.warning("Non-OpenAI provider without a tracing key; disabling tracing.")
        set_tracing_disabled(True)
