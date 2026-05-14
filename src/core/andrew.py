"""Shared agent core — event log, dispatch, and slash command handling.

Both the CLI (``AndrewCLI``) and the tray (``TrayController``) use this
class so the logic lives in one place.  Surfaces only need to override
the three output hooks and set ``self.domain`` before the event bus fires.
"""
from __future__ import annotations

import asyncio

from src.core.llm import ToolEvent, ToolResultEvent
from src.core.registry import registry
from src.core.server import server
from src.cli.filter import ThinkFilter


class AndrewCore:
    """Shared event-log, dispatch, and slash-command logic.

    Subclass or compose this and set ``self.domain`` before the event bus
    starts.  Override the three hooks to route output to your surface:

    ``_on_event_token(instance_id, token)``
        Called for every raw token (``str``, ``ToolEvent``,
        ``ToolResultEvent``) as it arrives.  The tray uses this to stream
        live to the panel; the CLI ignores it (prints atomically at the end).

    ``_on_event_output(instance_id, description, response)``
        Called once after a successful dispatch with the complete assembled
        response.  The CLI uses this to ``_bg_print`` the banner; the tray
        ignores it (already streamed via ``_on_event_token``).

    ``_on_event_done(instance_id)``
        Called unconditionally when dispatch ends (success, error, or
        cancellation).  The tray uses this to put the ``None`` sentinel into
        its token queue so Qt knows the stream is over; the CLI ignores it.
    """

    def __init__(self) -> None:
        self._event_log: dict[str, list[str]] = {}
        self._event_tool_log: dict[str, list[list[dict]]] = {}
        self._event_live: dict[str, list[str]] = {}
        self._event_live_tools: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Hooks — override in subclasses / surfaces
    # ------------------------------------------------------------------

    def _on_event_token(self, instance_id: str, token) -> None:
        pass

    def _on_event_output(self, instance_id: str, description: str, response: str) -> None:
        pass

    def _on_event_done(self, instance_id: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _event_dispatch(self, event) -> None:
        instance_id = getattr(event, "_instance_id", event.name)
        sid = getattr(event, "_bridge_sid", None)
        if sid:
            event._bridge_sid = None

        think_filter = ThinkFilter()
        parts: list[str] = []
        live = self._event_live[instance_id] = []
        live_tools = self._event_live_tools[instance_id] = []

        try:
            async for token in self.domain.generate_event(event.message):
                if isinstance(token, str):
                    for text, is_thinking in think_filter.process(token):
                        if not is_thinking:
                            parts.append(text)
                            live.append(text)
                    if sid:
                        server.put_token(sid, token)
                elif isinstance(token, ToolEvent) and token.tool_name:
                    live_tools.append({
                        "name": token.tool_name,
                        "args": token.tool_args or {},
                        "result": None,
                    })
                elif isinstance(token, ToolResultEvent):
                    for tc in reversed(live_tools):
                        if tc["name"] == token.tool_name and tc["result"] is None:
                            tc["result"] = token.result
                            break
                self._on_event_token(instance_id, token)
        except asyncio.CancelledError:
            self._event_live.pop(instance_id, None)
            self._event_live_tools.pop(instance_id, None)
            if sid:
                server.finish(sid, error="Event stopped")
            self._on_event_done(instance_id)
            raise
        except Exception as e:
            self._event_live.pop(instance_id, None)
            self._event_live_tools.pop(instance_id, None)
            if sid:
                server.finish(sid, error=str(e))
            self._on_event_done(instance_id)
            return
        else:
            if sid:
                server.finish(sid)

        tools_used = self._event_live_tools.pop(instance_id, [])
        self._event_live.pop(instance_id, None)
        response = "".join(parts).strip()
        log = self._event_log.setdefault(instance_id, [])
        log.append(response)
        tool_log = self._event_tool_log.setdefault(instance_id, [])
        tool_log.append(tools_used)
        if len(log) > 50:
            log.pop(0)
            tool_log.pop(0)

        self._on_event_output(instance_id, event.description, response)
        self._on_event_done(instance_id)

    # ------------------------------------------------------------------
    # Slash command handling  (/events  /stop  /status)
    # ------------------------------------------------------------------

    def handle_slash(self, cmd: str, bus) -> str | None:
        """Handle built-in slash commands.  Returns a response string, or
        ``None`` if the command is not recognised here (caller should try
        event-start parsing).
        """
        if cmd == "/events":
            return registry.list_commands(bus.running())

        if cmd.startswith("/stop"):
            parts = cmd.split(None, 1)
            if len(parts) == 1:
                running = bus.running()
                return ("Running: " + ", ".join(running)) if running else "No events running."
            key = parts[1]
            if bus.remove(key):
                for store in (
                    self._event_log, self._event_tool_log,
                    self._event_live, self._event_live_tools,
                ):
                    if key in store:
                        del store[key]
                    else:
                        for iid in [k for k in list(store) if k.rsplit("#", 1)[0] == key]:
                            del store[iid]
                return f"✓ Event '{key}' stopped"
            return f"No running event named '{key}'"

        if cmd.startswith("/status"):
            return self._format_status(cmd, bus)

        return None

    def _format_status(self, cmd: str, bus) -> str:
        parts = cmd.split(None, 1)
        running_ids = set(bus.running())

        if len(parts) == 1:
            all_ids = set(self._event_log) | set(self._event_live) | running_ids
            if not all_ids:
                return "No events recorded yet."
            lines = []
            for iid in sorted(all_ids):
                entries = self._event_log.get(iid, [])
                if iid in self._event_live:
                    status = "generating"
                elif iid in running_ids:
                    status = "running"
                else:
                    status = "stopped"
                lines.append(f"{iid} [{status}]: {len(entries)} iteration(s)")
            return "\n".join(lines)

        iid = parts[1]
        entries = self._event_log.get(iid, [])
        tool_entries = self._event_tool_log.get(iid, [])
        live_text = "".join(self._event_live.get(iid, [])).strip()
        live_tools = self._event_live_tools.get(iid, [])
        is_known = entries or iid in self._event_live or iid in running_ids
        if not is_known:
            return f"No event found for '{iid}'."

        header = f"=== {iid} — {len(entries)} iteration(s) ==="
        blocks: list[str] = []
        for i, (out, tools) in enumerate(
            zip(entries, tool_entries + [[]] * len(entries)), 1
        ):
            block = [f"[{i}]"]
            for tc in tools:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                result = tc["result"] or ""
                preview = result[:120] + "…" if len(result) > 120 else result
                block.append(f"  ↳ {tc['name']}({args_str})")
                if preview:
                    block.append(f"    {preview}")
            block.append(out if out else "(no output)")
            blocks.append("\n".join(block))

        if iid in self._event_live:
            block = ["[generating]"]
            for tc in live_tools:
                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                block.append(f"  ↳ {tc['name']}({args_str})")
                if tc["result"] is not None:
                    preview = tc["result"][:120] + "…" if len(tc["result"]) > 120 else tc["result"]
                    block.append(f"    {preview}")
                else:
                    block.append("    (running…)")
            block.append(live_text if live_text else "…")
            blocks.append("\n".join(block))
        elif not entries and iid in running_ids:
            blocks.append("[waiting for next trigger]")

        return header + "\n" + "\n\n".join(blocks)
