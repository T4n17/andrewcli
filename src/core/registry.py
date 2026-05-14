"""Unified registry — auto-discovery of domains, events, tools, and skills.

All four pluggable abstractions follow the same convention: drop a file
in the right directory and it is picked up at import time, no manual
registration required. Everything lives under the user's runtime config
directory at ``~/.config/andrewcli/`` (see :mod:`src.shared.paths`),
which is added to ``sys.path`` so ``domains.<name>`` and
``events.<name>`` resolve as regular Python packages.

* **Domains** — every subdirectory of ``domains/`` is a domain. Settings
  come from ``domains/<name>/config.yaml`` (overrides the global
  ``config.yaml``). :py:meth:`Registry.load_domain` returns a configured
  :class:`~src.core.domain.Domain` instance.
* **Events** — every ``events/*.py`` file is imported and any
  concrete :class:`~src.core.event.Event` subclass with a ``name``
  attribute is registered. :py:meth:`Registry.parse_slash_command` and
  :py:meth:`Registry.list_commands` turn user-typed ``/name [args]``
  strings into the right :class:`Event` instance (used identically by
  the CLI, the tray, and the HTTP server).
* **Tools** — every ``*.py`` module inside a domain's ``tools/``
  package contributes its concrete :class:`~src.core.tool.Tool`
  subclasses (excluding :class:`~src.core.skill.Skill`).
* **Skills** — every ``*.md`` file inside a domain's ``skills/`` folder
  is loaded as a :class:`Skill` instance built from its frontmatter.

Most consumers just use the module-level :data:`registry` singleton::

    from src.core.registry import registry

    for name in registry.domains():
        ...
    domain = registry.load_domain("general")
"""
from __future__ import annotations

import importlib
import inspect
import logging
import shlex
from pathlib import Path
from typing import get_type_hints

from src.core.event import Event
from src.core.skill import Skill
from src.core.tool import Tool
from src.shared.paths import DOMAINS_DIR, EVENTS_DIR

log = logging.getLogger(__name__)


class Registry:
    """Auto-discovery hub for domains, events, tools, and skills.

    State is intentionally minimal — just the two root directories the
    registry scans. The class form mirrors the other core abstractions
    (:class:`Tool`, :class:`Skill`, :class:`Event`, :class:`Domain`,
    :class:`Memory`, :class:`LLM`, :class:`ToolRouter`) so every piece
    of plug-in machinery follows the same shape.
    """

    def __init__(
        self,
        domains_dir: Path | str = DOMAINS_DIR,
        events_dir:  Path | str = EVENTS_DIR,
    ):
        self.domains_dir = Path(domains_dir)
        self.events_dir  = Path(events_dir)

    # ------------------------------------------------------------------
    # Domains
    # ------------------------------------------------------------------

    def domains(self) -> list[str]:
        """Return the sorted list of domain folder names under ``domains/``.

        A folder counts as a domain when it does not start with ``_`` or
        ``.`` and contains either a ``config.yaml``, a ``system_prompt.md``,
        or a ``tools``/``skills`` subdirectory. ``__init__.py`` is *not*
        required — domains are pure config folders.
        """
        if not self.domains_dir.is_dir():
            return []
        return sorted(
            p.name for p in self.domains_dir.iterdir()
            if p.is_dir()
            and not p.name.startswith("_")
            and not p.name.startswith(".")
            and (
                (p / "config.yaml").is_file()
                or (p / "system_prompt.md").is_file()
                or (p / "tools").is_dir()
                or (p / "skills").is_dir()
            )
        )

    def load_domain(self, name: str):
        """Return a :class:`~src.core.domain.Domain` configured for *name*.

        The folder must exist under :attr:`domains_dir`. Settings come
        from ``domains/<name>/config.yaml`` (optional) layered on top of
        the global ``config.yaml``. Any ``tools/`` or ``skills/``
        subfolders are auto-discovered by the :class:`Domain` constructor.
        """
        # Lazy import: ``domain`` imports back from this module, so a
        # top-level import would form a cycle.
        from src.core.domain import Domain
        if name not in self.domains():
            raise ValueError(
                f"Could not load domain '{name}': not found in {self.domains_dir}"
            )
        return Domain(name)

    # ------------------------------------------------------------------
    # Events  (+ slash command parsing)
    # ------------------------------------------------------------------

    def events(self) -> dict[str, type[Event]]:
        """Return ``{event.name: EventClass}`` for every concrete Event in ``events/``."""
        registry: dict[str, type[Event]] = {}
        if not self.events_dir.is_dir():
            return registry
        for path in sorted(self.events_dir.glob("*.py")):
            if path.stem == "__init__":
                continue
            try:
                module = importlib.import_module(f"events.{path.stem}")
            except Exception:
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, Event)
                    and obj is not Event
                    and isinstance(getattr(obj, "name", None), str)
                ):
                    registry[obj.name] = obj
        return registry

    @staticmethod
    def _instantiate_event(cls: type[Event], args_str: str) -> Event:
        """Instantiate an Event class from a raw args string.

        Tokenises args_str with shlex (so quoted strings work), coerces
        each token to its annotated type, and passes them positionally.
        Extra tokens beyond the declared params are ignored; missing
        optional params use their defaults.
        """
        sig = inspect.signature(cls.__init__)
        params = [p for name, p in sig.parameters.items() if name != "self"]

        if not params or not args_str.strip():
            return cls()

        try:
            tokens = shlex.split(args_str)
        except ValueError:
            tokens = args_str.split()

        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}

        args = []
        for param, token in zip(params, tokens):
            ann = hints.get(param.name, str)
            if ann is float:
                try:
                    token = float(token)
                except ValueError:
                    pass
            elif ann is int:
                try:
                    token = int(token)
                except ValueError:
                    pass
            args.append(token)

        return cls(*args)

    def parse_slash_command(self, text: str) -> Event | None:
        """Parse a ``/name [args]`` string into a matching Event instance.

        Returns ``None`` if *text* does not start with ``/`` or the event
        name is not in the registry. The caller should check for the
        ``/events`` help sentinel before calling this.
        """
        if not text.startswith("/"):
            return None
        parts = text[1:].strip().split(None, 1)
        if not parts:
            return None
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        cls = self.events().get(name)
        if cls is None:
            return None
        return self._instantiate_event(cls, args)

    def list_builtins(self) -> str:
        """Return the static built-in slash command reference (for ``/help``)."""
        return (
            "Built-in commands:\n"
            "- /help                        — show this help\n"
            "- /events                      — list available events and which are running\n"
            "- /stop [name|id]              — stop a running event by name or instance id\n"
            "- /status                      — list all events with status and iteration count\n"
            "- /status [id]                 — show full output log for a specific event\n"
            "- /clear                       — clear the screen (text only, memory kept)\n"
            "- /reset                       — clear conversation memory (text kept)\n"
        )

    def list_events(self, running: list[str] | None = None) -> str:
        """Return available events and which are currently running (for ``/events``).

        If *running* is provided (from :py:meth:`EventBus.running`),
        active events are shown with a marker so the user knows what
        can be stopped.
        """
        ev = self.events()
        lines: list[str] = []

        if running is not None:
            if running:
                lines.append("Running events: " + ", ".join(running))
                lines.append("  /stop [name|id]  — stop a running event")
            else:
                lines.append("No events currently running.")
            lines.append("")

        if not ev:
            lines.append("No events registered in events/.\n")
            return "\n".join(lines)

        # running contains instance IDs like "loop#1"; extract bare names for marker
        running_names = {iid.rsplit("#", 1)[0] for iid in (running or [])}
        lines.append("Available events:\n")
        for name, cls in sorted(ev.items()):
            sig = inspect.signature(cls.__init__)
            params = [n for n in sig.parameters if n != "self"]
            arg_hint = " " + " ".join(f"[{p}]" for p in params) if params else ""
            marker = " ●" if running is not None and name in running_names else ""
            lines.append(f"- /{name}{arg_hint}{marker}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tools  (per-domain auto-discovery)
    # ------------------------------------------------------------------

    def tools(self, package: str) -> list[Tool]:
        """Discover and instantiate all Tool subclasses in *package*.

        *package* is a dotted import path such as
        ``"domains.general.tools"``. Missing packages return an empty
        list so domains that don't declare tools Just Work.
        """
        try:
            pkg = importlib.import_module(package)
        except ModuleNotFoundError:
            return []

        pkg_file = getattr(pkg, "__file__", None)
        if not pkg_file:
            return []
        pkg_path = Path(pkg_file).parent

        tools: list[Tool] = []
        for path in sorted(pkg_path.glob("*.py")):
            if path.stem == "__init__":
                continue
            module_name = f"{package}.{path.stem}"
            try:
                module = importlib.import_module(module_name)
            except Exception:
                log.exception("failed to import tool module %s", module_name)
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, Tool):
                    continue
                if obj is Tool or obj is Skill or issubclass(obj, Skill):
                    continue
                # Only classes actually defined in this module, to avoid
                # re-instantiating tools that were merely imported.
                if obj.__module__ != module.__name__:
                    continue
                if inspect.isabstract(obj):
                    continue
                try:
                    tools.append(obj())
                except Exception:
                    log.exception("failed to instantiate tool %s", obj.__name__)
        return tools

    # ------------------------------------------------------------------
    # Skills  (per-domain auto-discovery)
    # ------------------------------------------------------------------

    def skills(self, skills_dir: Path | str) -> list[Skill]:
        """Instantiate a :class:`Skill` for every ``*.md`` file in *skills_dir*.

        Missing directories return an empty list so domains that don't
        declare skills Just Work.
        """
        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            return []

        skills: list[Skill] = []
        for path in sorted(skills_dir.glob("*.md")):
            try:
                skills.append(Skill(path))
            except Exception:
                log.exception("failed to load skill %s", path)
        return skills


# Default singleton wired to the user's runtime config directories.
# Most callers just import this rather than constructing their own
# ``Registry``; tests and tools can still instantiate the class directly
# with custom directories.
registry = Registry()
