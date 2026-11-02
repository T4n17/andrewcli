"""Shared agent core — event log, dispatch, and slash command handling.

Both the CLI (``AndrewCLI``) and the tray (``TrayController``) use this
class so the logic lives in one place.  Surfaces only need to override
the three output hooks and set ``self.domain`` before the event bus fires.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from src.core.automation import AutomationEvent
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
        Called when a visible dispatch ends (success, error, or cancellation).
        The tray uses this to put the ``None`` sentinel into its token queue so
        Qt knows the stream is over; the CLI ignores it.

    ``_on_loop(fn)``
        Runs *fn* on the asyncio loop that owns the event bus and returns
        its result. The default calls it directly; the tray, whose slash
        commands arrive on the Qt thread, replaces it.
    """

    def __init__(self) -> None:
        self._event_log: dict[str, list[str]] = {}
        self._event_tool_log: dict[str, list[list[dict]]] = {}
        self._event_live: dict[str, list[str]] = {}
        self._event_live_tools: dict[str, list[dict]] = {}
        # Automations already armed once this session; their startup
        # triggers do not fire again on domain switches or reloads.
        self._automations_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Hooks — override in subclasses / surfaces
    # ------------------------------------------------------------------

    def _on_event_token(self, instance_id: str, token) -> None:
        pass

    def _on_event_output(self, instance_id: str, description: str, response: str) -> None:
        pass

    def _on_event_done(self, instance_id: str) -> None:
        pass

    def _on_loop(self, fn):
        return fn()

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _event_dispatch(self, event) -> None:
        instance_id = getattr(event, "_instance_id", event.name)
        sid = getattr(event, "_bridge_sid", None)
        if sid:
            event._bridge_sid = None

        is_silent = getattr(event, "silent", False)
        think_filter = ThinkFilter()
        parts: list[str] = []
        live = self._event_live[instance_id] = []
        live_tools = self._event_live_tools[instance_id] = []

        command = getattr(event, "command", "")
        if command:
            stream = self._command_stream(command)
        else:
            stream = self.domain.generate_event(
                event.message, getattr(event, "system_message", ""),
                required_tools=getattr(event, "required_tools", []),
            )

        try:
            async for token in stream:
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
                if not is_silent:
                    self._on_event_token(instance_id, token)
        except asyncio.CancelledError:
            self._event_live.pop(instance_id, None)
            self._event_live_tools.pop(instance_id, None)
            if sid:
                server.finish(sid, error="Event stopped")
            if not is_silent:
                self._on_event_done(instance_id)
            raise
        except Exception as e:
            self._event_live.pop(instance_id, None)
            self._event_live_tools.pop(instance_id, None)
            if sid:
                server.finish(sid, error=str(e))
            if not is_silent:
                self._on_event_done(instance_id)
            return
        else:
            if sid:
                server.finish(sid)

        tools_used = self._event_live_tools.pop(instance_id, [])
        self._event_live.pop(instance_id, None)
        response = "".join(parts).strip()
        if response and hasattr(event, "on_response"):
            try:
                event.on_response(response)
            except Exception:
                pass
        log = self._event_log.setdefault(instance_id, [])
        log.append(response)
        tool_log = self._event_tool_log.setdefault(instance_id, [])
        tool_log.append(tools_used)
        if len(log) > 50:
            log.pop(0)
            tool_log.pop(0)

        if not is_silent:
            self._on_event_output(instance_id, event.description, response)
            self._on_event_done(instance_id)

    async def _command_stream(self, cmd: str):
        """Run a slash command on behalf of an event and stream its output.

        Resolution order matches typed input: built-in commands, then the
        domain's workflows, then events (which are started on the bus).
        """
        response = self.handle_slash(cmd, self.domain.event_bus)
        if response is not None:
            yield response
            return
        workflow = self.domain.run_workflow(cmd)
        if workflow is not None:
            async for token in workflow:
                yield token
            return
        try:
            event = registry.parse_slash_command(cmd)
        except ValueError as exc:
            yield str(exc)
            return
        if event is None:
            yield f"Unknown command: {cmd}"
            return
        instance_id = self.domain.event_bus.add(event)
        yield f"✓ Event '{event.name}' started [{instance_id}]"

    # ------------------------------------------------------------------
    # Automations
    # ------------------------------------------------------------------

    def arm_automations(self) -> tuple[list[str], list[str]]:
        """Start the enabled automations that apply to the active domain.

        Must run on the event bus loop. Startup triggers fire only the
        first time an automation is armed in this session. Returns
        ``(instance_ids, load_errors)``.
        """
        bus = self.domain.event_bus
        automations, errors = registry.automations()
        armed: list[str] = []
        for automation in automations:
            if not automation.enabled or not automation.applies_to(self.domain.name):
                continue
            first = automation.name not in self._automations_seen
            self._automations_seen.add(automation.name)
            run_now = first and automation.runs_at_startup
            if not run_now and not automation.is_scheduled:
                continue
            armed.append(bus.add(AutomationEvent(automation, run_now=run_now)))
        return armed, errors

    def _disarm_automations(self) -> None:
        bus = self.domain.event_bus
        for instance_id, event in bus.events().items():
            if isinstance(event, AutomationEvent):
                bus.remove(instance_id)

    def _handle_automations(self, cmd: str) -> str:
        parts = cmd.split(None, 2)
        action = parts[1] if len(parts) > 1 else ""

        if action == "reload":
            def _reload():
                self._disarm_automations()
                return self.arm_automations()
            armed, errors = self._on_loop(_reload)
            lines = [f"✓ {len(armed)} automation(s) armed" + (f": {', '.join(armed)}" if armed else "")]
            lines += [f"  ✗ {error}" for error in errors]
            return "\n".join(lines)

        if action == "run":
            if len(parts) < 3:
                return "Usage: /automations run [name]"
            automations, _ = registry.automations()
            automation = next((a for a in automations if a.name == parts[2].strip()), None)
            if automation is None:
                return f"No automation named '{parts[2].strip()}'"
            instance_id = self._on_loop(
                lambda: self.domain.event_bus.add(AutomationEvent(automation, oneshot=True))
            )
            return f"✓ Automation '{automation.name}' started [{instance_id}]"

        if action:
            return "Usage: /automations | /automations reload | /automations run [name]"
        return self._format_automations()

    def _format_automations(self) -> str:
        automations, errors = registry.automations()
        if not automations and not errors:
            return f"No automations defined. Add *.yaml files to {registry.automations_dir}"
        armed = {
            event.name: event for event in self.domain.event_bus.events().values()
            if isinstance(event, AutomationEvent)
        }
        now = datetime.now()
        lines = ["Automations:\n"]
        for automation in sorted(automations, key=lambda a: a.name):
            event = armed.get(automation.name)
            if not automation.enabled:
                state = "disabled"
            elif not automation.applies_to(self.domain.name):
                state = f"inactive in this domain (domain: {', '.join(automation.domains)})"
            elif event is None:
                state = "done" if automation.name in self._automations_seen else "not armed"
            elif event.next_run is None or event.next_run <= now:
                state = "running"
            else:
                state = f"next run {event.next_run:%Y-%m-%d %H:%M}"
            lines.append(
                f"- {automation.name} [{state}] — {automation.describe_trigger()} → {automation.run}\n"
            )
        if errors:
            lines.append("Errors:\n")
            lines += [f"- {error}\n" for error in errors]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Slash command handling  (/events  /stop  /status)
    # ------------------------------------------------------------------

    def handle_slash(self, cmd: str, bus) -> str | None:
        """Handle built-in slash commands.  Returns a response string, or
        ``None`` if the command is not recognised here (caller should try
        event-start parsing).
        """
        if cmd == "/clear":
            return "✓ Screen cleared."

        if cmd == "/reset":
            self.domain.llm.memory.clear()
            return "✓ Memory cleared."

        if cmd == "/help":
            return registry.list_builtins()

        if cmd == "/events":
            return registry.list_events(bus.running())

        if cmd == "/workflows":
            self.domain.load_workflows()
            return registry.list_workflows(self.domain.workflows)

        if cmd == "/skills":
            return registry.list_skills(self.domain.name)

        if cmd == "/automations" or cmd.startswith("/automations "):
            return self._handle_automations(cmd)

        if cmd in ("/rag", "/rag status"):
            return self.domain.rag.status()

        if cmd == "/rag metrics":
            return self.domain.rag.metrics()

        if cmd == "/rag reindex":
            if self.domain.rag.request_refresh(force=True):
                return "✓ Knowledge base reindex started."
            return "Knowledge base indexer is not running."

        if cmd.startswith("/rag"):
            return "Usage: /rag status | /rag metrics | /rag reindex"

        if cmd.startswith("/stop"):
            parts = cmd.split(None, 1)
            if len(parts) == 1:
                running = bus.running()
                return ("Running: " + ", ".join(running)) if running else "No events running."
            key = parts[1]
            if self._on_loop(lambda: bus.remove(key)):
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
