import asyncio
import hashlib
import logging
from pathlib import Path

import yaml

from src.core.llm import LLM, RouteEvent
from src.core.memory import Memory
from src.core.rag import KnowledgeBase
from src.core.registry import registry
from src.core.router import ToolRouter
from src.shared.config import Config
from src.shared.paths import DOMAINS_DIR, RAG_CACHE_DIR

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
    * ``knowledgebase/``   — optional, documents searchable through the
                             knowledge tools when a turn needs them.

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

        self.load_workflows()
        self.router = ToolRouter(self.llm.client, self.llm.model)
        cfg = Config()
        self.rag = KnowledgeBase(
            self._domain_dir / "knowledgebase",
            cache_path=self._rag_cache_path(),
            enabled=cfg.rag_enabled,
            reranker_model=cfg.rag_reranker_model,
            rerank_candidates=cfg.rag_rerank_candidates,
            embedding_model=cfg.rag_embedding_model,
            ann_threshold=cfg.rag_ann_threshold,
            ann_shard_size=cfg.rag_ann_shard_size,
            watch=cfg.rag_watch,
        )
        try:
            self.rag.start()
        except RuntimeError:
            pass
        # Single "agent busy" lock. Both user turns (generate) and event
        # dispatches (generate_event) acquire it, so:
        #   - only one interaction streams to the UI at a time
        #   - events queue FIFO behind each other and behind user turns
        #   - CLI and tray inherit identical behavior without per-surface
        #     locks
        # asyncio.Lock guarantees FIFO wake-up of waiters. AndrewCore hands
        # the lock to the next domain on a switch. Events live on
        # AndrewCore's bus, not here, so they survive domain switches.
        self.busy_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _rag_cache_path(self) -> Path:
        """Return the knowledge-base cache, keyed by the domain folder.

        Two config trees with a same-named domain must never share (and
        re-sync) one index, so the path includes a digest of the folder.
        """
        digest = hashlib.sha256(str(self._domain_dir).encode()).hexdigest()[:10]
        return RAG_CACHE_DIR / f"{self.name}-{digest}.sqlite3"

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

        Memory is left untouched.
        """
        domain_dir = self._domain_dir

        # Re-discover tools and skills from disk; the registry re-imports
        # modules whose files changed.
        self.tools = registry.tools(f"domains.{self.name}.tools")
        self.skills = registry.skills(domain_dir / "skills")

        # Reload system prompt if the file changed.
        new_prompt = self._load_system_prompt(domain_dir)
        if new_prompt != self.system_prompt:
            self.system_prompt = new_prompt
            self.llm.set_system_prompt(new_prompt)

        # Re-read the global and per-domain config.yaml so model/api/routing
        # edits take effect on the next turn without a restart.
        old_api, old_model = self.api_base_url, self.model
        Config().reload()
        self._apply_config(self._load_domain_config())
        if self.api_base_url != old_api or self.model != old_model:
            self.llm = LLM(
                api_base_url=self.api_base_url,
                model=self.model,
                memory=self.llm.memory,
            )
            self.llm.set_system_prompt(self.system_prompt)
            self.router = ToolRouter(self.llm.client, self.llm.model)

        # Last, so workflows get the current client, model, and tools.
        self.load_workflows()

    def load_workflows(self) -> None:
        """Re-discover workflows and bind them to the current LLM and tools."""
        self.workflows = registry.workflows(f"domains.{self.name}.workflows")
        for wf in self.workflows:
            wf.configure(self.llm.client, self.llm.model, self.tools)

    def close(self) -> None:
        self.rag.stop()

    # ------------------------------------------------------------------
    # Workflows
    # ------------------------------------------------------------------

    def run_workflow(self, cmd: str):
        """Parse ``/name [args]`` and return a streaming async generator.

        Returns ``None`` when no workflow matches *cmd*. When a workflow
        is found the returned generator acquires :attr:`busy_lock` before
        delegating to :meth:`~src.core.workflow.Workflow.run`, so workflow
        execution is serialised with user turns and event dispatches.
        """
        self.load_workflows()
        result = registry.parse_workflow_command(self.workflows, cmd)
        if result is None:
            return None
        wf, kwargs = result
        return self._run_workflow_gen(wf, kwargs)

    async def _run_workflow_gen(self, wf, kwargs: dict):
        async with self.busy_lock:
            async for token in wf.run(**kwargs):
                yield token

    async def generate_event(
        self,
        prompt: str,
        system_message: str = "",
        required_tools: list[str] | None = None,
        only_tools: list[str] | None = None,
    ):
        """One-shot generation for event dispatches.

        Routes to the right tools like generate() does, but uses a fresh LLM
        with no conversation context so event exchanges never pollute the
        conversation memory or affect routing for user queries.

        If `system_message` is provided it is appended to the domain system
        prompt so event instructions carry system-level authority rather than
        being sent as a user message.

        If `only_tools` is given, the model gets exactly those tools and no
        skills. Use it when the prompt carries untrusted text, so that text
        cannot steer the model into running other tools.

        Serialized via `busy_lock` so events queue behind any in-flight
        user turn or earlier event.
        """
        async with self.busy_lock:
            route_text = system_message or prompt
            if only_tools is not None:
                tools, skills = [t for t in self.tools if t.name in only_tools], []
            elif self.routing_enabled:
                tools, skills = await self.router.route(route_text, self.tools, self.skills)
            else:
                tools, skills = list(self.tools), list(self.skills)

            existing_names = {t.name for t in tools}
            required_names = {name for s in skills for name in s.required_tools}
            if required_tools:
                required_names.update(required_tools)
            for tool in self.tools:
                if tool.name in required_names and tool.name not in existing_names:
                    tools.append(tool)
                    existing_names.add(tool.name)

            # A disabled Memory keeps the run isolated: it neither sees the
            # conversation summary nor writes the exchange to memory.json.
            event_llm = LLM(
                api_base_url=self.llm.api_base_url,
                model=self.llm.model,
                memory=Memory(enabled=False),
            )
            if system_message:
                combined = "\n\n".join(filter(None, [self.system_prompt, system_message]))
                event_llm.set_system_prompt(combined)
            else:
                event_llm.set_system_prompt(self.system_prompt)
            yield RouteEvent([item.name for item in tools + skills])
            user_prompt = prompt or "Execute."
            async for token in event_llm.generate(user_prompt, tools, skills):
                yield token

    async def generate(self, prompt: str):
        # Serialized via `busy_lock`: blocks any concurrent user turn or
        # event dispatch until this generator is fully consumed (or
        # closed early via aclose()). See `busy_lock` docstring.
        async with self.busy_lock:
            self.reload()
            # The knowledge base is only consulted through its tools, so the
            # model decides per turn whether the request needs the documents.
            knowledge_tools = self.rag.agent_tools()
            context = self.rag.agent_guidance() if knowledge_tools else ""
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
            for tool in knowledge_tools:
                if tool.name not in existing_names:
                    tools.append(tool)
                    existing_names.add(tool.name)

            yield RouteEvent([item.name for item in tools + skills])
            async for token in self.llm.generate(prompt, tools, skills, context=context):
                yield token
