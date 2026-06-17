"""
TR: Agentlarin kullanabilecegi fonksiyon toollari.
    `@function_tool` dekoratoru; fonksiyon imzasindan JSON schema'yi, docstring'den
    aciklamalari otomatik uretir. Gercek projede burada DB/HTTP cagrilari olur.
EN: Function tools the agents can use.
    The `@function_tool` decorator auto-generates the JSON schema from the function
    signature and descriptions from the docstring. In a real project this is where
    your DB/HTTP calls would live.
"""

from __future__ import annotations

import logging
from typing import Annotated

from agents import RunContextWrapper, function_tool

from app.ai.context import AppContext

logger = logging.getLogger(__name__)


@function_tool
async def get_weather(
    ctx: RunContextWrapper[AppContext],
    city: Annotated[str, "The city to look up the weather for, e.g. 'Istanbul'."],
) -> str:
    """Return the current weather for a given city.

    TR: Gercek hayatta burada bir hava durumu API'sine istek atilir.
    EN: In real life this would call a weather API.
    """
    logger.info("get_weather called (user=%s, city=%s)", ctx.context.user_id, city)
    # TR: Demo amacli sabit yanit. / EN: Stubbed response for the demo.
    return f"The weather in {city} is 22°C and sunny."


@function_tool
async def search_knowledge_base(
    ctx: RunContextWrapper[AppContext],
    query: Annotated[str, "What the user wants to know about the product."],
) -> str:
    """Search the product knowledge base and return the most relevant snippet.

    TR: Gercek hayatta bu bir vektor arama / RAG cagrisidir.
    EN: In real life this would be a vector search / RAG call.
    """
    logger.info("search_knowledge_base called (user=%s, query=%s)", ctx.context.user_id, query)
    # TR: Kucuk, sahte bir bilgi tabani. / EN: A tiny, fake knowledge base.
    knowledge = {
        "refund": "Refunds are processed within 5-7 business days to the original payment method.",
        "password": "Reset your password from Settings > Security > Reset password.",
        "plan": "We offer Free, Pro, and Enterprise plans. Pro unlocks unlimited runs.",
    }
    for keyword, answer in knowledge.items():
        if keyword in query.lower():
            return answer
    return "NO_MATCH"  # TR: Agent bunu gorup durust cevap verir. / EN: The agent sees this and answers honestly.


# TR: Tek noktadan import edebilmek icin. / EN: For single-point imports.
ALL_TOOLS = [get_weather, search_knowledge_base]
