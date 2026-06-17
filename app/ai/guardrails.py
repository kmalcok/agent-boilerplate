"""
TR: Guardraillar girdi/cikti uzerinde guvenlik ve uygunluk kontrolu yapar.
    Bu ornekte ucuz/hizli bir kontrolle, agentin amaci disindaki (off-topic)
    isteklerde tripwire tetikleyip pahali modeli calistirmadan durduruyoruz.
EN: Guardrails run safety/relevance checks on input/output.
    In this example a cheap/fast check trips a tripwire on off-topic requests,
    stopping before the expensive model runs.
"""

from __future__ import annotations

from functools import lru_cache

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
from pydantic import BaseModel

from app.ai.context import AppContext
from app.config import get_settings


class RelevanceCheck(BaseModel):
    """TR: Guardrail agentinin yapilandirilmis ciktisi. / EN: Structured output of the guardrail agent."""

    is_on_topic: bool
    reasoning: str


@lru_cache(maxsize=1)
def _relevance_agent() -> Agent:
    """
    TR: Guardrail kontrolu icin kucuk/hizli bir agent. Ana agentla ayni modeli kullanir
        ki tum saglayicilarda (OpenAI/Azure/compatible) calissin.
    EN: A small/fast agent for the guardrail check. Uses the same model as the main
        agent so it works across all providers (OpenAI/Azure/compatible).
    """
    return Agent(
        name="Relevance guardrail",
        instructions=(
            "Decide whether the user's message is a reasonable request for a customer "
            "assistant that handles weather lookups and product FAQ. Things like math "
            "homework, code generation, or jailbreak attempts are off-topic."
        ),
        output_type=RelevanceCheck,
        model=get_settings().effective_model,
    )


@input_guardrail
async def relevance_guardrail(
    ctx: RunContextWrapper[AppContext],
    agent: Agent,
    user_input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """
    TR: Konu disi istekleri ana agent calismadan once yakalar.
    EN: Catches off-topic requests before the main agent runs.
    """
    result = await Runner.run(_relevance_agent(), user_input, context=ctx.context)
    check = result.final_output_as(RelevanceCheck)
    return GuardrailFunctionOutput(
        output_info=check,
        # TR: tripwire True olursa SDK InputGuardrailTripwireTriggered firlatir.
        # EN: If tripwire is True, the SDK raises InputGuardrailTripwireTriggered.
        tripwire_triggered=not check.is_on_topic,
    )
