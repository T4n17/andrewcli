"""Discover Event subclasses and parse /name [args] slash commands.

The same parsing logic is used by the CLI, tray, and server so a
command typed in any surface works identically.

Slash command syntax:
    /name [arg]

    /timer 10            → TimerEvent(10.0)
    /file war_news/news.md → FileEvent("war_news/news.md")
    /project Build a web app → ProjectEvent("Build a web app")
    /agent_loop Research AI trends → AgentLoopEvent("Research AI trends")
    /events              → print available commands (no event started)
"""
import importlib
import inspect
from typing import get_type_hints

from src.core.event import Event
from src.shared.paths import EVENTS_DIR


def available_events() -> dict[str, type[Event]]:
    """Return {event.name: EventClass} for every concrete Event in src/events/."""
    registry: dict[str, type[Event]] = {}
    if not EVENTS_DIR.is_dir():
        return registry
    for path in sorted(EVENTS_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        try:
            module = importlib.import_module(f"src.events.{path.stem}")
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


def _instantiate(cls: type[Event], args_str: str) -> Event:
    """Instantiate an Event class from a raw args string.

    Tokenises args_str with shlex (so quoted strings work), coerces each
    token to its annotated type, and passes them positionally. Extra tokens
    beyond the declared params are ignored; missing optional params use
    their defaults.
    """
    import shlex

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


def parse_slash_command(text: str) -> Event | None:
    """Parse a '/name [args]' string and return the matching Event instance.

    Returns None if text does not start with '/' or the event name is
    not in the registry. The caller should check for '/events' (the
    help sentinel) before calling this.
    """
    if not text.startswith("/"):
        return None
    parts = text[1:].strip().split(None, 1)
    if not parts:
        return None
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    cls = available_events().get(name)
    if cls is None:
        return None
    return _instantiate(cls, args)


def list_commands(running: list[str] | None = None) -> str:
    """Return a human-readable table of available slash commands.

    If `running` is provided (from EventBus.running()), active events
    are shown with a marker so the user knows what can be stopped.
    """
    registry = available_events()
    lines = []

    if running is not None:
        if running:
            lines.append("Running events: " + ", ".join(running))
            lines.append("  /stop [name]         — stop a running event")
        else:
            lines.append("No events currently running.")
        lines.append("")

    if not registry:
        lines.append("No events registered in src/events/.")
        return "\n".join(lines)

    lines.append("Available slash commands:")
    for name, cls in sorted(registry.items()):
        sig = inspect.signature(cls.__init__)
        params = [n for n in sig.parameters if n != "self"]
        arg_hint = " " + " ".join(f"[{p}]" for p in params) if params else ""
        marker = " ●" if running is not None and name in running else ""
        lines.append(f"  /{name}{arg_hint}{marker}")
    lines.append("  /events              — show this list")
    lines.append("  /stop [name]         — stop a running event")
    return "\n".join(lines)
