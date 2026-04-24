import asyncio
import logging
from abc import ABC
from typing import List

from src.core.event import EventBus
from src.core.llm import LLM, RouteEvent
from src.core.router import EmbeddingRouter, ToolRouter
from src.shared.config import Config

log = logging.getLogger(__name__)


def _make_router():
    """Pick the routing backend based on config, with safe fallback.

    - "embed": fastembed-based cosine similarity (default, fast).
    - "llm":   classic LLM-as-classifier (slower, more flexible).

    If fastembed is requested but not installed, we fall back to the
    LLM router rather than crashing the whole app.
    """
    cfg = Config()
    backend = getattr(cfg, "router_backend", "embed")
    if backend == "embed":
        try:
            return EmbeddingRouter(threshold=getattr(cfg, "router_threshold", None))
        except ImportError as exc:
            log.warning(
                "router_backend='embed' requested but fastembed is not "
                "installed (%s); falling back to the LLM router",
                exc,
            )
    return ToolRouter()


class Domain(ABC):
    system_prompt: str = ""
    tools: List = []
    skills: List = []
    events: List = []
    # When False, skip the router entirely and expose every declared
    # tool/skill to the LLM on every turn. Useful for domains where the
    # full toolset is always relevant (e.g. coding).
    routing_enabled: bool = True

    def __init__(self):
        # Copy class-level mutable attributes onto the instance so that
        # two Domain instances (or subclasses declared with shared lists)
        # cannot accidentally mutate each other's tool/skill/event state.
        self.tools = list(self.tools)
        self.skills = list(self.skills)
        self.events = list(self.events)
        self.llm = LLM()
        self.llm.set_system_prompt(self.system_prompt)
        self.router = _make_router()
        self.event_bus = EventBus(self.events)

        # Single "agent busy" lock. Both user turns (generate) and event
        # dispatches (generate_event) acquire it, so:
        #   - only one interaction streams to the UI at a time
        #   - events queue FIFO behind each other and behind user turns
        #   - CLI and tray inherit identical behavior without per-surface
        #     locks
        # asyncio.Lock guarantees FIFO wake-up of waiters.
        self.busy_lock = asyncio.Lock()

        # Embedding router benefits from background warm-up: downloads
        # the model and pre-embeds the catalog while the user is still
        # reading the UI, so the first real route() is instant.
        warm = getattr(self.router, "warm", None)
        if callable(warm):
            warm(self.tools, self.skills)

    async def generate_event(self, prompt: str):
        """One-shot generation for event dispatches.

        Routes to the right tools like generate() does, but uses a fresh LLM
        with no conversation context so event exchanges never pollute the
        conversation memory or affect routing for user queries.

        Serialized via `busy_lock` so events queue behind any in-flight
        user turn or earlier event.
        """
        async with self.busy_lock:
            if self.routing_enabled:
                tools, skills = await self.router.route(prompt, self.tools, self.skills)
            else:
                tools, skills = list(self.tools), list(self.skills)

            existing_names = {t.name for t in tools}
            required_names = {name for s in skills for name in s.required_tools}
            for tool in self.tools:
                if tool.name in required_names and tool.name not in existing_names:
                    tools.append(tool)
                    existing_names.add(tool.name)

            event_llm = LLM()
            event_llm.set_system_prompt(self.system_prompt)
            yield RouteEvent([item.name for item in tools + skills])
            async for token in event_llm.generate(prompt, tools, skills):
                yield token

    async def generate(self, prompt: str):
        # Serialized via `busy_lock`: blocks any concurrent user turn or
        # event dispatch until this generator is fully consumed (or
        # closed early via aclose()). See `busy_lock` docstring.
        async with self.busy_lock:
            if self.routing_enabled:
                tools, skills = await self.router.route(
                    prompt, self.tools, self.skills,
                    summary=self.llm.memory.summary,
                    last_exchange=self.llm.memory.last_exchange,
                )
            else:
                tools, skills = list(self.tools), list(self.skills)

            # Pull in any tools declared as required by the selected skills
            existing_names = {t.name for t in tools}
            required_names = {name for s in skills for name in s.required_tools}
            for tool in self.tools:
                if tool.name in required_names and tool.name not in existing_names:
                    tools.append(tool)
                    existing_names.add(tool.name)

            yield RouteEvent([item.name for item in tools + skills])
            async for token in self.llm.generate(prompt, tools, skills):
                yield token
