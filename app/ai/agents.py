"""
TR: Agent tanimlari. Burada bir triage (yonlendirme) agenti, iki uzman agent
    (weather + faq) ve aralarindaki handoff'lar kurulur.
EN: Agent definitions. Here we set up a triage (routing) agent, two specialist
    agents (weather + faq), and the handoffs between them.

TR: Tasarim deseni: "Handoff" (merkezi olmayan yonlendirme). Triage agenti
    konuyu anlar ve kontrolu uzman agente devreder.
EN: Design pattern: "Handoff" (decentralized routing). The triage agent
    understands the topic and hands control to a specialist agent.
"""

from __future__ import annotations

from functools import lru_cache

from agents import Agent

from app.ai.context import AppContext
from app.ai.guardrails import relevance_guardrail
from app.ai.instructions import load_instructions
from app.ai.tools import get_weather, search_knowledge_base
from app.config import Settings


@lru_cache(maxsize=1)
def build_triage_agent_cached(model: str) -> Agent[AppContext]:
    """
    TR: Agent grafini bir kez kurup onbellekler (model adina gore).
        Talimatlar dosyadan, toollar tools.py'den, guardrail guardrails.py'den gelir.
    EN: Builds the agent graph once and caches it (keyed by model name).
        Instructions come from files, tools from tools.py, guardrail from guardrails.py.
    """
    weather_agent = Agent[AppContext](
        name="Weather agent",
        handoff_description="Answers questions about current weather in a city.",
        instructions=load_instructions("weather"),
        tools=[get_weather],
        model=model,
    )

    faq_agent = Agent[AppContext](
        name="FAQ agent",
        handoff_description="Answers product, account, and billing questions.",
        instructions=load_instructions("faq"),
        tools=[search_knowledge_base],
        model=model,
    )

    triage_agent = Agent[AppContext](
        name="Triage agent",
        instructions=load_instructions("triage"),
        # TR: Triage konuyu anlayip uzmanlara devreder. / EN: Triage understands the topic and delegates to specialists.
        handoffs=[weather_agent, faq_agent],
        # TR: Girdi guardrail'i sadece zincirin ILK agentinda calisir. / EN: Input guardrail runs only on the FIRST agent.
        input_guardrails=[relevance_guardrail],
        model=model,
    )
    return triage_agent


def build_triage_agent(settings: Settings) -> Agent[AppContext]:
    """TR: Ayarlardan etkin modeli alarak agent grafini dondurur. / EN: Returns the agent graph using the effective model."""
    return build_triage_agent_cached(settings.effective_model)
