"""
TR: Uygulama ile SDK arasindaki ince servis katmani. Web sunucusu ve CLI bu
    katmani cagirir; SDK detaylari (Runner, session, guardrail hatalari) burada saklanir.
EN: A thin service layer between the application and the SDK. The web server and CLI
    call into this layer; SDK details (Runner, session, guardrail errors) are hidden here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    Runner,
    RunConfig,
)
from openai.types.responses import ResponseTextDeltaEvent

from app.ai.agents import build_triage_agent
from app.ai.context import AppContext
from app.ai.hooks import UsageLoggingHooks, usage_to_dict
from app.ai.provider import configure_provider
from app.ai.sessions import build_session
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class AgentReply:
    """TR: Servis cevabi. / EN: Service reply."""

    output: str
    last_agent: str
    blocked: bool = False  # TR: Guardrail engelledi mi? / EN: Did a guardrail block it?
    # TR: Bu run'in token kullanimi (reasoning ve cached detaylariyla).
    # EN: Token usage for this run (with reasoning and cached details).
    usage: dict[str, int] = field(default_factory=dict)


class AgentService:
    """
    TR: Tum istekler icin paylasilan tekil servis. Acilista provider'i yapilandirir
        ve agent grafini hazirlar.
    EN: A single shared service for all requests. Configures the provider at startup
        and prepares the agent graph.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        # TR: Provider'i bir kez global olarak ayarla (OpenAI/Azure/compatible).
        # EN: Configure the provider once, globally (OpenAI/Azure/compatible).
        configure_provider(self.settings)
        self.agent: Agent[AppContext] = build_triage_agent(self.settings)
        # TR: Token kullanimini loglayan stateless hook; tum run'larda yeniden kullanilir.
        # EN: Stateless hook that logs token usage; reused across all runs.
        self.hooks = UsageLoggingHooks()

    def _run_config(self) -> RunConfig:
        # TR: Tracing'de gorunecek workflow adi vb. / EN: Workflow name shown in tracing, etc.
        return RunConfig(workflow_name=self.settings.workflow_name)

    async def chat(
        self,
        message: str,
        *,
        session_id: str,
        context: AppContext | None = None,
    ) -> AgentReply:
        """
        TR: Tek seferlik (non-streaming) bir konusma turu calistirir.
        EN: Runs a single (non-streaming) conversation turn.
        """
        ctx = context or AppContext(user_id=session_id)
        session = build_session(session_id, self.settings)
        try:
            result = await Runner.run(
                self.agent,
                message,
                context=ctx,
                session=session,
                hooks=self.hooks,
                run_config=self._run_config(),
            )
            # TR: Token kullanimini RunResult uzerinden context_wrapper.usage ile cekiyoruz.
            # EN: We pull token usage from the RunResult via context_wrapper.usage.
            return AgentReply(
                output=str(result.final_output),
                last_agent=result.last_agent.name,
                usage=usage_to_dict(result.context_wrapper.usage),
            )
        except InputGuardrailTripwireTriggered:
            # TR: Konu disi/uygunsuz istek; nazik bir red mesaji dondururuz.
            # EN: Off-topic/inappropriate request; return a polite refusal.
            logger.info("Input guardrail tripped (session=%s)", session_id)
            return AgentReply(
                output="I can only help with weather lookups and product questions.",
                last_agent=self.agent.name,
                blocked=True,
            )
        except MaxTurnsExceeded:
            logger.warning("Max turns exceeded (session=%s)", session_id)
            return AgentReply(
                output="This request got too complex. Please try narrowing it down.",
                last_agent=self.agent.name,
            )

    async def chat_stream(
        self,
        message: str,
        *,
        session_id: str,
        context: AppContext | None = None,
    ) -> AsyncIterator[str]:
        """
        TR: Token token (streaming) cevap uretir. Web tarafinda SSE icin kullanilir.
        EN: Yields the answer token by token (streaming). Used for SSE on the web side.
        """
        ctx = context or AppContext(user_id=session_id)
        session = build_session(session_id, self.settings)
        result = Runner.run_streamed(
            self.agent,
            message,
            context=ctx,
            session=session,
            hooks=self.hooks,
            run_config=self._run_config(),
        )
        try:
            async for event in result.stream_events():
                # TR: Sadece ham metin delta'larini disari verelim. / EN: Emit only raw text deltas.
                if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                    yield event.data.delta
            # TR: Stream bittikten sonra toplam usage netlesir; burada loglariz.
            # EN: Usage totals settle after the stream finishes; we log them here.
            logger.info("stream usage (session=%s): %s", session_id, usage_to_dict(result.context_wrapper.usage))
        except InputGuardrailTripwireTriggered:
            yield "I can only help with weather lookups and product questions."
