"""
TR: Lifecycle hook'lari. SDK, calismanin belirli anlarinda (agent baslangic/bitis,
    LLM cagrisi, tool cagrisi, handoff) bu callback'leri tetikler. Burada token
    kullanimini (usage) loglamak icin kullaniyoruz.
EN: Lifecycle hooks. The SDK fires these callbacks at specific moments of a run
    (agent start/end, LLM call, tool call, handoff). Here we use them to log token
    usage.

TR: Onemli: reasoning token'lari output_tokens'in bir ALT KUMESIDIR; toplama tekrar
    eklemeyin. cached_tokens da input_tokens'in alt kumesidir.
EN: Note: reasoning tokens are a SUBSET of output_tokens; do not add them back to the
    total. cached_tokens is likewise a subset of input_tokens.
"""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent, RunContextWrapper, RunHooks
from agents.usage import Usage

from app.ai.context import AppContext

logger = logging.getLogger(__name__)


def usage_to_dict(usage: Usage) -> dict[str, int]:
    """
    TR: `Usage` objesini loglama/JSON icin duz bir dict'e cevirir.
    EN: Flattens a `Usage` object into a plain dict for logging/JSON.
    """
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        # TR: input_tokens icindeki cache'lenmis kisim. / EN: cached portion within input_tokens.
        "cached_input_tokens": usage.input_tokens_details.cached_tokens,
        # TR: output_tokens icindeki reasoning kismi. / EN: reasoning portion within output_tokens.
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
    }


class UsageLoggingHooks(RunHooks[AppContext]):
    """
    TR: Her agent bittiginde o ana kadar biriken token kullanimini loglar.
        `context.usage`, calisma boyunca (tool'lar + handoff'lar dahil) aggregate edilir.
    EN: Logs the accumulated token usage whenever an agent finishes.
        `context.usage` is aggregated across the whole run (including tools + handoffs).
    """

    async def on_agent_end(
        self,
        context: RunContextWrapper[AppContext],
        agent: Agent[AppContext],
        output: Any,
    ) -> None:
        u = context.usage
        logger.info(
            "usage[%s] user=%s requests=%d input=%d output=%d (reasoning=%d) cached=%d total=%d",
            agent.name,
            context.context.user_id,
            u.requests,
            u.input_tokens,
            u.output_tokens,
            u.output_tokens_details.reasoning_tokens,
            u.input_tokens_details.cached_tokens,
            u.total_tokens,
        )
