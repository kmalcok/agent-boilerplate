"""
TR: Terminal uzerinden interaktif demo. SDK kurulumunu hizlica denemek icin idealdir.
    Calistirma: `python -m app.cli` ya da `agent-cli`.
EN: An interactive terminal demo. Ideal for quickly trying out the SDK setup.
    Run: `python -m app.cli` or `agent-cli`.
"""

from __future__ import annotations

import asyncio
import logging

from app.ai.context import AppContext
from app.ai.service import AgentService
from app.config import get_settings


async def _chat_loop() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    service = AgentService(settings)

    print(f"Agent Boilerplate (provider={settings.model_provider.value}, model={settings.effective_model})")
    print("TR: Cikmak icin 'exit'. / EN: Type 'exit' to quit.\n")

    session_id = "cli-session"
    context = AppContext(user_id=session_id)

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue

        # TR: Streaming ile token token yazdir. / EN: Print token by token via streaming.
        print("Assistant: ", end="", flush=True)
        async for delta in service.chat_stream(message, session_id=session_id, context=context):
            print(delta, end="", flush=True)
        print("\n")


def main() -> None:
    """TR: `agent-cli` giris noktasi. / EN: `agent-cli` entry point."""
    asyncio.run(_chat_loop())


if __name__ == "__main__":
    main()
