"""
TR: Production HTTP API (FastAPI). Iki uc nokta sunar:
    - POST /chat        : tek seferlik cevap (JSON)
    - POST /chat/stream : token token cevap (SSE)
    Calistirma: `uvicorn app.server:app --host 0.0.0.0 --port 8080`
EN: Production HTTP API (FastAPI). Exposes two endpoints:
    - POST /chat        : single response (JSON)
    - POST /chat/stream : token-by-token response (SSE)
    Run: `uvicorn app.server:app --host 0.0.0.0 --port 8080`
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.ai.context import AppContext
from app.ai.service import AgentService
from app.config import get_settings

logger = logging.getLogger(__name__)

# TR: Servis tekil; lifespan icinde olusturulur. / EN: The service is a singleton; created in lifespan.
_service: AgentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """TR: Acilista provider'i ayarla ve servisi kur. / EN: Configure provider and build the service at startup."""
    global _service
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    _service = AgentService(settings)
    logger.info("AgentService ready (provider=%s).", settings.model_provider.value)
    yield
    # TR: Burada kapanis temizligi yapilabilir. / EN: Shutdown cleanup can go here.


app = FastAPI(title="Agent Boilerplate", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    """TR: Sohbet istek govdesi. / EN: Chat request body."""

    message: str = Field(..., description="The user's message.")
    session_id: str = Field("default", description="Conversation/user id for memory.")
    is_pro_user: bool = False


def _service_or_raise() -> AgentService:
    if _service is None:  # pragma: no cover - guarded by lifespan
        raise RuntimeError("Service not initialized.")
    return _service


@app.get("/health")
async def health() -> dict[str, str]:
    """TR: Saglik kontrolu (k8s/probe icin). / EN: Health check (for k8s/probes)."""
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, object]:
    """TR: Tek seferlik cevap. / EN: Single response."""
    service = _service_or_raise()
    ctx = AppContext(user_id=req.session_id, is_pro_user=req.is_pro_user)
    reply = await service.chat(req.message, session_id=req.session_id, context=ctx)
    return {
        "output": reply.output,
        "last_agent": reply.last_agent,
        "blocked": reply.blocked,
        # TR: Token kullanimi (reasoning ve cached detaylariyla). / EN: Token usage (with reasoning and cached details).
        "usage": reply.usage,
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """TR: Server-Sent Events ile streaming cevap. / EN: Streaming response via Server-Sent Events."""
    service = _service_or_raise()
    ctx = AppContext(user_id=req.session_id, is_pro_user=req.is_pro_user)

    async def event_generator():
        async for delta in service.chat_stream(req.message, session_id=req.session_id, context=ctx):
            # TR: SSE format: "data: <metin>\n\n". / EN: SSE format: "data: <text>\n\n".
            yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
