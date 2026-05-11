import asyncio
import inspect
import logging
from abc import ABC
from pathlib import Path

from src.core.event import EventBus
from src.core.llm import LLM, RouteEvent
from src.core.registry import available_skills, available_tools
from src.core.router import ToolRouter

log = logging.getLogger(__name__)


class Domain(ABC):
    # Fallback system prompt, used only when the domain folder doesn't
    # ship a ``system_prompt.md``. New domains should prefer the file.
    system_prompt: str = ""
    model: str = None
    api_base_url: str = None
    # When False, skip the router entirely and expose every declared
    # tool/skill to the LLM on every turn. Useful for domains where the
    # full toolset is always relevant (e.g. coding).
    routing_enabled: bool = True

    def __init__(self):
        # Autodiscover tools, skills, and the system prompt from this
        # domain's own package folder. A subclass lives in
        # ``domains/<name>/domain.py``, so:
        #   - tools         come from ``domains/<name>/tools/*.py``
        #   - skills        come from ``domains/<name>/skills/*.md``
        #   - system_prompt comes from ``domains/<name>/system_prompt.md``
        #     (falls back to the class-level ``system_prompt`` string if
        #     no file is present, so quick experiments keep working).
        domain_pkg = type(self).__module__.rsplit(".", 1)[0]
        domain_file = inspect.getfile(type(self))
        domain_dir = Path(domain_file).resolve().parent

        self.tools = available_tools(f"{domain_pkg}.tools")
        self.skills = available_skills(domain_dir / "skills")
        self.system_prompt = self._load_system_prompt(domain_dir)

        self.llm = LLM(api_base_url=self.api_base_url, model=self.model)
        self.llm.set_system_prompt(self.system_prompt)
        self.router = ToolRouter(
            api_base_url=self.llm.api_base_url,
            model=self.llm.model,
        )
        # Events are independent of domains — they're registered
        # dynamically via slash commands (``/timer 30`` etc.). The bus
        # starts empty and picks them up through ``EventBus.add()``.
        self.event_bus = EventBus()

        # Single "agent busy" lock. Both user turns (generate) and event
        # dispatches (generate_event) acquire it, so:
        #   - only one interaction streams to the UI at a time
        #   - events queue FIFO behind each other and behind user turns
        #   - CLI and tray inherit identical behavior without per-surface
        #     locks
        # asyncio.Lock guarantees FIFO wake-up of waiters.
        self.busy_lock = asyncio.Lock()

    def _load_system_prompt(self, domain_dir: Path) -> str:
        """Return the prompt shipped in ``<domain_dir>/system_prompt.md``.

        Falls back to the class-level ``system_prompt`` string when the
        file is missing, so ad-hoc in-code domains still work. Trailing
        whitespace is stripped so downstream prompt formatting doesn't
        have to worry about a dangling newline.
        """
        prompt_file = domain_dir / "system_prompt.md"
        if prompt_file.is_file():
            try:
                return prompt_file.read_text().strip()
            except Exception:
                log.exception("failed to read %s", prompt_file)
        return self.system_prompt

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

            event_llm = LLM(api_base_url=self.llm.api_base_url, model=self.llm.model)
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
