import asyncio
import importlib
import logging
import sys
from pathlib import Path

import yaml

from src.core.event import EventBus
from src.core.llm import LLM, RouteEvent
from src.core.registry import registry
from src.core.router import ToolRouter
from src.shared.paths import DOMAINS_DIR

log = logging.getLogger(__name__)


# Keys a per-domain ``config.yaml`` may set to override the global config
# when this domain is active. Anything outside this set is ignored at the
# Domain layer (it may still be honored by other consumers if they
# explicitly read the per-domain file).
_DOMAIN_OVERRIDE_KEYS = ("api_base_url", "model", "routing_enabled")


class Domain:
    """Runtime-configurable agent domain.

    A domain is just a folder under ``~/.config/andrewcli/domains/<name>/``
    containing:

    * ``config.yaml``      — optional, overrides global settings
                             (``api_base_url``, ``model``,
                             ``routing_enabled``). Missing or empty
                             means "inherit everything from global".
    * ``system_prompt.md`` — optional, system prompt text.
    * ``tools/*.py``       — optional, :class:`Tool` subclasses.
    * ``skills/*.md``      — optional, frontmatter-driven skills.

    There is no longer a per-domain Python class: the same ``Domain``
    instance is reused for every domain, parametrised by ``name``.
    """

    # Defaults applied when neither the per-domain ``config.yaml`` nor
    # the global ``config.yaml`` set a value. ``None`` for url/model
    # leaves :class:`LLM` to fall back to its own built-in defaults.
    api_base_url: str | None = None
    model: str | None = None
    routing_enabled: bool = True
    system_prompt: str = ""

    def __init__(self, name: str):
        self.name = name
        self._domain_dir = (DOMAINS_DIR / name).resolve()
        if not self._domain_dir.is_dir():
            raise ValueError(f"Domain '{name}' not found at {self._domain_dir}")

        self._apply_config(self._load_domain_config())
        self.system_prompt = self._load_system_prompt(self._domain_dir)

        self.tools = registry.tools(f"domains.{name}.tools")
        self.skills = registry.skills(self._domain_dir / "skills")

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

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_domain_config(self) -> dict:
        """Return the parsed per-domain ``config.yaml`` (empty if absent)."""
        path = self._domain_dir / "config.yaml"
        if not path.is_file():
            return {}
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            log.exception("failed to read %s", path)
            return {}

    def _apply_config(self, domain_cfg: dict) -> None:
        """Apply the per-domain config on top of the global Config defaults.

        Per-domain values win; otherwise the same key on the global
        :class:`~src.shared.config.Config` is consulted; otherwise the
        class-level default sticks.
        """
        from src.shared.config import Config  # local import: avoid cycles
        global_cfg = Config()
        for key in _DOMAIN_OVERRIDE_KEYS:
            if key in domain_cfg and domain_cfg[key] is not None:
                setattr(self, key, domain_cfg[key])
            elif hasattr(global_cfg, key):
                setattr(self, key, getattr(global_cfg, key))
            # else: keep the class-level default

    def _load_system_prompt(self, domain_dir: Path) -> str:
        """Return the prompt shipped in ``<domain_dir>/system_prompt.md``.

        Falls back to the class-level ``system_prompt`` string when the
        file is missing. Trailing whitespace is stripped so downstream
        prompt formatting doesn't have to worry about a dangling
        newline.
        """
        prompt_file = domain_dir / "system_prompt.md"
        if prompt_file.is_file():
            try:
                return prompt_file.read_text().strip()
            except Exception:
                log.exception("failed to read %s", prompt_file)
        return self.system_prompt

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Reload tools, skills, system prompt, and config from disk.

        Called before each user turn so runtime edits take effect immediately:
        add/remove/edit a skill file, edit a tool module, flip routing_enabled,
        change api_base_url or model in config.yaml, update system_prompt.md.

        Memory and the event bus are left untouched.
        """
        domain_dir = self._domain_dir
        tools_pkg = f"domains.{self.name}.tools"

        # Reload all already-imported tool modules so in-place edits are picked up.
        try:
            pkg_mod = sys.modules.get(tools_pkg)
            if pkg_mod and getattr(pkg_mod, "__file__", None):
                pkg_path = Path(pkg_mod.__file__).parent
                for path in sorted(pkg_path.glob("*.py")):
                    if path.stem == "__init__":
                        continue
                    mod_name = f"{tools_pkg}.{path.stem}"
                    if mod_name in sys.modules:
                        importlib.reload(sys.modules[mod_name])
        except Exception:
            log.exception("domain reload: failed to reload tool modules")

        # Re-discover tools and skills from disk (picks up added/removed files).
        self.tools = registry.tools(tools_pkg)
        self.skills = registry.skills(domain_dir / "skills")

        # Reload system prompt if the file changed.
        new_prompt = self._load_system_prompt(domain_dir)
        if new_prompt != self.system_prompt:
            self.system_prompt = new_prompt
            self.llm.set_system_prompt(new_prompt)

        # Re-read the per-domain config.yaml so model/api/routing edits
        # take effect on the next turn without a restart.
        old_api, old_model = self.api_base_url, self.model
        self._apply_config(self._load_domain_config())
        if self.api_base_url != old_api or self.model != old_model:
            old_memory = self.llm.memory
            self.llm = LLM(api_base_url=self.api_base_url, model=self.model)
            self.llm.memory = old_memory
            self.llm.set_system_prompt(self.system_prompt)
            self.router = ToolRouter(
                api_base_url=self.llm.api_base_url,
                model=self.llm.model,
            )

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
            self.reload()
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
